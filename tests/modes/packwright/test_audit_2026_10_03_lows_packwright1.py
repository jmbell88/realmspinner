"""Regression tests closing the 2026-10-03 audit's Low Packwright findings
(batch packwright1: packwright-10 .. packwright-16).

Each test's name is the claim. Where a test could only fail before the fix
because a helper did not exist, the test's docstring says so.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import layout as lay
from realmspinner.studio.modes.packwright.engine import rpack
from realmspinner.studio.modes.packwright.engine.document import PackDoc
from realmspinner.studio.modes.packwright.engine.layout import PackSettings
from realmspinner.studio.modes.packwright.engine.sources import (
    Sprite,
    SpriteMeta,
    sprite_meta,
)
from realmspinner.studio.modes.packwright.ui.panes import preview as packwright_preview
from realmspinner.studio.modes.packwright.ui.panes import sources as packwright_sources

from .test_packwright_mode import FakeCtx, _Done, _pack, _sprite, _tab
from .test_packwright_sources import _Spy, _Texture
from .test_rpack import _doc, _rewrite


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


@pytest.fixture(autouse=True)
def _no_slice_cache_leak():
    packwright_sources._slice_grid_cache = None
    yield
    packwright_sources._slice_grid_cache = None


# --- packwright-10 ------------------------------------------------------------


def test_a_refused_repack_submit_does_not_clear_packing_while_a_pack_is_in_flight():
    """A pack is running; the user touches a setting; the runner refuses the
    second submit because the first key is still in flight. ``packing`` belongs
    to the running pack and the refusal must leave it True."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    tab.packing = True  # the pack already in flight
    tab.pack_dirty = True

    packwright_mode.request_pack(ctx, tab)

    assert ctx.submitted == [f"packwright-pack:{tab.uid}"]
    assert tab.pack_dirty is True, "the edit must stay armed for the next pump"
    assert tab.packing is True, "a refused submit said 'not packing' about a running pack"


def test_a_refused_first_submit_still_reads_not_packing():
    """The other half: with nothing in flight a refusal restores the False the
    tab had, so the old guard test's claim still holds."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    assert tab.packing is False
    packwright_mode.request_pack(ctx, tab)
    assert tab.packing is False


# --- packwright-11 ------------------------------------------------------------


def test_outlines_skips_frames_outside_the_visible_region(ui):
    """The pane pushes a clip rect for the atlas box; ``_outlines`` used to draw
    every packed frame regardless, so zooming into one corner of a large atlas
    still paid for every off-screen frame."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=64)
    _pack(ctx, tab)
    assert tab.layout is not None and len(tab.layout.frames) == 64
    tab.view.zoom = 1.0
    tab.view.pan = (0.0, 0.0)
    state = SimpleNamespace(selected=None)

    ui.new_frame()
    ui.begin("host")
    real = ui.get_window_draw_list()
    spy = _Spy(real)
    # Exactly the corner of the atlas that holds a handful of frames.
    real.push_clip_rect((100.0, 100.0), (120.0, 120.0), False)
    try:
        packwright_preview._outlines(state, tab, spy, (100.0, 100.0))
    finally:
        real.pop_clip_rect()
        ui.end()
        ui.end_frame()

    drawn = len(spy.of("add_rect"))
    assert 1 <= drawn < 64, f"{drawn} of 64 frame rectangles were drawn for a 20px window"


def test_outlines_still_draws_a_frame_whose_pivot_is_the_only_visible_part(ui):
    """Culling is on the frame rectangle with the pivot cross's own reach
    added, so a frame just off the edge whose cross pokes in is not lost."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    _pack(ctx, tab)
    frame = tab.layout.frames[0]
    tab.view.zoom = 1.0
    tab.view.pan = (0.0, 0.0)
    state = SimpleNamespace(selected=None)
    # The clip window sits just past the frame's right edge, closer than the
    # cross's arm is long.
    lo_x = 100.0 + frame.x + frame.w + 2.0
    ui.new_frame()
    ui.begin("host")
    real = ui.get_window_draw_list()
    spy = _Spy(real)
    real.push_clip_rect((lo_x, 100.0), (lo_x + 50.0, 300.0), False)
    # A pivot at the frame's right edge: its cross reaches into the window.
    pivoted = dataclasses.replace(frame, pivot=(float(frame.w + frame.trim[0]), 1.0))
    tab.layout = SimpleNamespace(frames=(pivoted,))
    try:
        packwright_preview._outlines(state, tab, spy, (100.0, 100.0))
    finally:
        real.pop_clip_rect()
        ui.end()
        ui.end_frame()
    assert spy.of("add_line"), "a pivot cross poking into the window was culled away"


# --- packwright-12 ------------------------------------------------------------


def test_the_slice_preview_draws_a_bounded_number_of_rects_whatever_the_cell_size(
    ui, monkeypatch
):
    """``_slice_preview`` submitted one imgui rect per grid cell with no cap, and
    ``input_int`` reports every keystroke, so an intermediate 1 x 1 cell on a
    large sheet drew a rect per pixel."""
    monkeypatch.setattr(packwright_sources, "_slice_texture", lambda ctx, pixels: _Texture())
    ctx = SimpleNamespace(viewer=None, state=SimpleNamespace(preview={}))
    sheet = np.zeros((256, 256, 4), dtype=np.uint8)  # 65,536 one-pixel cells

    ui.new_frame()
    ui.begin("host")
    real = ui.get_window_draw_list
    spy: list[_Spy] = []

    def spied():
        if not spy:
            spy.append(_Spy(real()))
        return spy[0]

    ui.get_window_draw_list = spied
    try:
        packwright_sources._slice_preview(ctx, sheet, (1, 1))
    finally:
        ui.get_window_draw_list = real
        ui.end()
        ui.end_frame()

    marks = len(spy[0].of("add_rect")) + len(spy[0].of("add_rect_filled"))
    assert marks <= 4096, f"{marks} per-cell rects were submitted in one frame"


# --- packwright-13 ------------------------------------------------------------


def test_a_power_of_two_maxrects_pack_is_never_a_non_power_of_two_atlas():
    """Twelve 300 px sprites under a 1500 px ceiling: the search appended the
    non-power-of-two ``(1500, 1500)`` limit as its last candidate and
    ``power_of_two=True`` returned it unchanged."""
    sprites = [_square(f"s{i}", 300) for i in range(12)]
    settings = PackSettings(mode="maxrects", power_of_two=True, max_size=1500, padding=2)
    try:
        result = lay.maxrects_layout(sprites, settings)
    except ValueError as exc:
        # Refusing by name is the other legal answer.
        assert "power" in str(exc).lower()
        return
    for side in (result.width, result.height):
        assert side & (side - 1) == 0, f"{result.width}x{result.height} is not power-of-two"


def test_a_power_of_two_refusal_names_power_of_two_and_the_remedy():
    sprites = [_square(f"s{i}", 300) for i in range(12)]
    settings = PackSettings(mode="maxrects", power_of_two=True, max_size=1500, padding=2)
    with pytest.raises(ValueError, match="power-of-two") as caught:
        lay.maxrects_layout(sprites, settings)
    assert "max size" in str(caught.value)


def test_a_non_power_of_two_pack_may_still_use_the_limit_as_its_last_candidate():
    """The limit candidate stays for the packs it was written for."""
    sprites = [_square(f"s{i}", 300) for i in range(12)]
    settings = PackSettings(mode="maxrects", power_of_two=False, max_size=1500, padding=2)
    result = lay.maxrects_layout(sprites, settings)
    assert max(result.width, result.height) <= 1500


# --- packwright-14 ------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_set_pivot_refuses_a_non_finite_coordinate(bad):
    doc = PackDoc()
    doc.add_source(_sprite("a"))
    uid = doc.sources[0].uid
    with pytest.raises(ValueError, match="finite"):
        doc.set_pivot(uid, (bad, 1.0))
    with pytest.raises(ValueError, match="finite"):
        doc.set_pivot(uid, (1.0, bad))
    assert doc.sources[0].sprite.meta.pivot is None


def test_a_producer_pivot_that_is_not_finite_costs_the_pivot_not_the_sprite():
    meta = sprite_meta({"pivot": [math.inf, 1.0]})
    assert meta.pivot is None


def test_the_manifest_is_never_written_with_a_non_finite_token():
    """A sprite built around the door (frozen ``meta`` set directly) must fail at
    save, not write ``Infinity`` that ``read_rpack`` then refuses at reopen."""
    doc = PackDoc()
    base = _sprite("a")
    bad = Sprite(key="a", name="a", pixels=base.pixels, meta=SpriteMeta(pivot=(math.inf, 0.0)))
    doc.add_source(bad)
    with pytest.raises(ValueError):
        rpack.manifest_json(doc)


# --- packwright-15 ------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    ["../../evil\nxxx", "a/b", "a\\b", "tab\there", "x" * 65],
)
def test_read_rpack_refuses_a_name_override_that_rename_source_would_refuse(hostile):
    data = _rewrite(_doc(), lambda m: m["sources"][0].update(name_override=hostile))
    with pytest.raises(ValueError, match="name"):
        rpack.read_rpack(data)


@pytest.mark.parametrize("hostile", ["line\nbreak", "x" * 256])
def test_read_rpack_refuses_an_unbounded_display_name(hostile):
    data = _rewrite(_doc(), lambda m: m["sources"][0].update(name=hostile))
    with pytest.raises(ValueError, match="name"):
        rpack.read_rpack(data)


def test_read_rpack_keeps_a_display_name_with_a_slash_because_a_layer_may_hold_one():
    """The display name is what the sprite came in as (an Inker layer's name);
    refusing a ``/`` would make this app unable to reopen a file it wrote."""
    data = _rewrite(_doc(), lambda m: m["sources"][0].update(name="body/arm"))
    assert rpack.read_rpack(data).sources[0].sprite.name == "body/arm"


def test_read_rpack_still_opens_a_document_with_ordinary_names():
    doc = rpack.read_rpack(rpack.rpack_bytes(_doc()))
    assert [s.name_override for s in doc.sources].count("hero") == 1


# --- packwright-16 ------------------------------------------------------------


def test_dropping_several_images_raises_one_toast_for_the_batch(tmp_path):
    """pygame raises one ``DROPFILE`` per file; the shell collects a pump's
    drops and hands Packwright the list, so twenty files are one task, one
    "Added N sprite(s)." toast and one undo step.

    Fails before the fix because ``EventsMixin._on_drops`` (the batch door)
    did not exist: ``_events`` called ``_on_drop`` once per file."""
    from PIL import Image

    from realmspinner.studio.shell.events import EventsMixin

    paths = []
    for index in range(3):
        path = tmp_path / f"tile{index}.png"
        Image.new("RGBA", (4, 4), (index * 40, 0, 0, 255)).save(path)
        paths.append(path)

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    ctx.state.mode = "packwright"
    shell = SimpleNamespace(app_ctx=ctx)
    shell._on_drop = lambda path: EventsMixin._on_drop(shell, path)

    EventsMixin._on_drops(shell, paths)

    assert len(ctx.submitted) == 1, f"{len(ctx.submitted)} tasks for one drop"
    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[0], ctx.result))
    assert len(tab.doc.sources) == 3
    added = [message for message, _kind in ctx.toasts if message.startswith("Added")]
    assert len(added) == 1, added
    tab.doc.undo()
    assert tab.doc.sources == [], "one Ctrl+Z must undo the whole drop"


def test_a_mixed_drop_still_toasts_the_unsupported_file_once(tmp_path):
    from PIL import Image

    from realmspinner.studio.shell.events import EventsMixin

    png = tmp_path / "a.png"
    Image.new("RGBA", (4, 4), (9, 9, 9, 255)).save(png)
    txt = tmp_path / "notes.txt"
    txt.write_text("x")

    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    ctx.state.mode = "packwright"
    shell = SimpleNamespace(app_ctx=ctx)
    shell._on_drop = lambda path: EventsMixin._on_drop(shell, path)

    EventsMixin._on_drops(shell, [png, txt])

    refusals = [m for m, kind in ctx.toasts if kind == "error"]
    assert len(refusals) == 1 and "Packwright opens" in refusals[0]
    assert len(ctx.submitted) == 1


def _square(key: str, side: int) -> Sprite:
    pixels = np.full((side, side, 4), 200, dtype=np.uint8)
    return Sprite(key=key, name=key, pixels=pixels)

