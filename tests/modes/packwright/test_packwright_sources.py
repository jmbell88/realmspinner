"""Packwright's sources pane: the rename field and the tile-set import popup.

No such module existed before the 2026-09-07 audit. The first row below is a
source-inspection check in ``test_undo_gesture_doors.py``'s fifth-section
style, because the field it covers draws only inside a selected row's own
sub-tree, which a headless frame does not open on its own.

The second row (W3.3) draws in a real, rendererless imgui context instead --
``troupe/test_troupe_preview_draw.py``'s shape -- because what is asserted is
*which cells got which mark*, and a source-text check cannot tell a rect drawn
at the right cell from one drawn at the wrong one.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.studio.modes.packwright.ui.panes import sources as packwright_sources


def test_renaming_a_packwright_source_is_one_undo_step_not_one_per_keystroke():
    """packwright-03: ``_row``'s rename field called ``widgets.input_text``
    with no ``commit=True``, while ``clay_outliner.py``'s identical widget
    passes it -- so ``rename_source`` (an unconditional ``history.push``)
    fired once per keystroke. Typing "lead" pushed 4 steps, and one Ctrl+Z
    left "lea" rather than undoing the whole rename."""
    source = inspect.getsource(packwright_sources._row)
    after_field = source.split('"##rename"', 1)[1]
    call_end = after_field.index(")")
    assert "commit=True" in after_field[:call_end]


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    fixture there."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_tileset_popup_closed_does_not_scan_the_shared_preview_cache_every_frame(
    ui, monkeypatch
):
    """packwright-01 (2026-09-18 audit): with the tile-set popup closed and
    nothing parked, ``_tileset_popup`` ran ``clear_tileset_import`` outright
    on every frame -- which calls ``docmodes.release_prefix``, a scan of the
    whole app-wide ``ctx.state.preview`` dict, to forget a texture that was
    never parked in the first place."""
    calls: list[str] = []
    monkeypatch.setattr(
        packwright_sources.docmodes,
        "release_prefix",
        lambda ctx, prefix: calls.append(prefix),
    )
    state = SimpleNamespace(
        tileset_import=None,
        tileset_import_uid="",
        tileset_import_open=False,
        tileset_preview_key=None,
    )
    ctx = SimpleNamespace(state=SimpleNamespace(preview={}))
    ui.new_frame()
    ui.begin("host")
    try:
        for _ in range(3):
            packwright_sources._tileset_popup(ctx, state)
    finally:
        ui.end()
        ui.end_frame()
    assert calls == [], "closed popup with nothing parked scanned the preview cache"


def test_add_buttons_grey_with_a_reason_while_saving():
    """packwright-02 (2026-09-18 audit): "Add an image..." and "Add a tile
    set..." greyed out while a tab was saving with no ``reason=`` passed to
    ``widgets.disabled_button``, so the tooltip that explains every other
    greyed control in this app said nothing here."""
    source = inspect.getsource(packwright_sources.draw)
    # ``index("):")`` rather than ``index(")")``: each call also carries a
    # ``(-1, 0)`` size tuple, whose own closing paren is not followed by a
    # colon, so it does not end the search early the way a bare ``)`` would.
    add_image = source.split("Add an image...", 1)[1]
    call_end = add_image.index("):")
    assert "reason=widgets.DOCUMENT_SAVING_WHY" in add_image[:call_end]

    add_tileset = source.split("Add a tile set...", 1)[1]
    call_end = add_tileset.index("):")
    assert "reason=widgets.DOCUMENT_SAVING_WHY" in add_tileset[:call_end]


class _Spy:
    """The window draw list, recording every call and forwarding all of them.

    ``troupe/test_troupe_preview_draw.py``'s ``_Spy`` verbatim: a real
    ``imgui`` rejects an argument list a fake would silently accept.
    """

    def __init__(self, real):
        self.real = real
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        target = getattr(self.real, name)

        def recorded(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return target(*args, **kwargs)

        return recorded

    def of(self, name: str) -> list[tuple]:
        return [args for called, args, _kw in self.calls if called == name]


class _Texture:
    """Enough of a moderngl texture for ``widgets.texture_ref``: there is no
    renderer in this context, so nothing is registered and only ``glo`` is
    ever read."""

    glo = 1


def _sheet() -> np.ndarray:
    """A 9 x 10 sheet, tile 4 x 4 -- two whole rows and columns of cells, a
    1px remainder on the right and a 2px remainder on the bottom. Cells
    (0, 0) and (1, 0) carry opaque pixels; (0, 1) and (1, 1) are left fully
    transparent, so the grid ``tileset_occupancy`` returns is a checkerboard
    of exactly two kept and two dropped cells."""
    pixels = np.zeros((10, 9, 4), dtype=np.uint8)
    pixels[0:4, 0:4, 3] = 255
    pixels[4:8, 0:4, 3] = 255
    return pixels


def _draw(ui, monkeypatch) -> _Spy:
    monkeypatch.setattr(
        packwright_sources, "_slice_texture", lambda ctx, pixels: _Texture()
    )
    ctx = SimpleNamespace(viewer=None, state=SimpleNamespace(preview={}))
    spy: list[_Spy] = []
    real = ui.get_window_draw_list

    def spied():
        if not spy:
            spy.append(_Spy(real()))
        return spy[0]

    ui.new_frame()
    ui.begin("host")
    ui.get_window_draw_list = spied
    try:
        packwright_sources._slice_preview(ctx, _sheet(), (4, 4))
    finally:
        ui.get_window_draw_list = real
        ui.end()
        ui.end_frame()
    assert spy, "_slice_preview never asked for the window draw list"
    return spy[0]


def test_the_slice_preview_marks_kept_dropped_and_remainder(ui, monkeypatch):
    """W3.3: the tile-set import popup said ``"2 x 2 cells - 2 tile(s), 2
    empty dropped"`` and nothing else -- no way to see *which* cells were
    which, or that a 1px strip on the right and a 2px strip on the bottom
    were never counted at all because ``tileset_occupancy`` leaves a
    less-than-one-tile edge out of its grid. The popup now draws that same
    grid over the sheet: kept cells outlined, dropped cells dimmed, the two
    remainder strips hatched."""
    draws = _draw(ui, monkeypatch)

    # Kept: two cells, outlined -- ``add_rect``, not filled, so the sheet
    # underneath a kept cell still shows through.
    assert len(draws.of("add_rect")) == 2
    # Dropped: two cells, dimmed -- filled rather than outlined, the opposite
    # mark from a kept cell's.
    assert len(draws.of("add_rect_filled")) == 2
    # The remainder: a right strip and a bottom strip both exist on this
    # sheet, and both are hatched with at least one diagonal line.
    assert len(draws.of("add_line")) >= 2
