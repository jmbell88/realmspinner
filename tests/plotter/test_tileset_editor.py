"""The tileset editor sheet, and the refusal that guards a removal.

The refusal *is* the feature. A gid is a firstgid plus a local id, so dropping
a tileset out from under painted cells does not clear them -- it renumbers what
they mean. And "it is still in use" is a sentence a user cannot act on, so the
message carries the count and the layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.kernels.grid2d.tileset import Tileset
from warlock.studio import plotter_state
from warlock.studio.plotter.tilemap import MapDoc


@pytest.fixture
def ui(monkeypatch):
    """The shared headless imgui context; see ``_ui_context`` for why this is
    a per-module fixture rather than a shared ``conftest`` one."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _tileset(name: str = "Overworld", tiles: int = 4) -> Tileset:
    pixels = np.zeros((8, 8 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, tile_w=8, tile_h=8, pixels=pixels)


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer(name="Ground")
    return doc


def test_an_unused_tileset_can_be_removed_and_the_step_undone():
    doc = _doc()
    doc.add_tileset(_tileset())
    head = doc.history.head
    doc.remove_tileset(0)
    assert doc.tilesets == []
    doc.undo()
    assert len(doc.tilesets) == 1 and doc.history.head == head


def test_a_used_tileset_is_refused_by_name_with_a_count_and_a_layer():
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    layer = doc.layers[0]
    data = layer.data.copy()
    data[0, :2] = ref.firstgid
    doc.write_region(layer.uid, 0, 0, data)
    with pytest.raises(ValueError) as raised:
        doc.remove_tileset(0)
    message = str(raised.value)
    assert "Overworld" in message and "Ground" in message
    assert "2" in message
    assert len(doc.tilesets) == 1, "and nothing was removed"


def test_a_survivor_keeps_its_firstgid_so_painted_cells_still_mean_what_they_meant():
    """A hole in gid space is legal -- wmap requires only that they increase."""

    doc = _doc()
    first = doc.add_tileset(_tileset("A"))
    second = doc.add_tileset(_tileset("B"))
    before = second.firstgid
    assert before > first.firstgid
    doc.remove_tileset(0)
    assert doc.tilesets[0].firstgid == before


def test_usage_counts_across_every_layer():
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.add_tile_layer(name="Detail")
    for layer in doc.layers:
        data = layer.data.copy()
        data[0, 0] = ref.firstgid
        doc.write_region(layer.uid, 0, 0, data)
    used, where = doc.tileset_usage(0)
    assert used == 2 and where in {"Ground", "Detail"}


def test_a_tileset_held_only_by_a_stamp_is_still_in_use():
    """The stamps are document state -- stored in the map, written to ``.wmap``
    since VERSION 11 -- and they hold gids exactly as a layer does. Counting
    only the layers let a tileset that nothing had painted yet but a stamp still
    named be removed; ``next_firstgid`` then reuses the range, and recalling the
    stamp paints the *new* tileset's tiles with no sign anything happened."""
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.set_stamp(1, np.full((1, 2), ref.firstgid, dtype=np.uint32), name="roof")

    used, where = doc.tileset_usage(0)
    assert used == 2
    assert "stamp 1" in where and "roof" in where
    with pytest.raises(ValueError, match="Overworld"):
        doc.remove_tileset(0)
    assert len(doc.tilesets) == 1


def test_reading_a_wmap_refuses_a_stamp_whose_tileset_is_gone():
    """``_validate`` checked the layers and the tile objects and not the stamps,
    so a map carrying a stamp nothing accounts for opened without complaint and
    only went wrong on recall -- where the user's gesture was a number key and
    there is nothing useful to say."""
    from warlock.studio.plotter import wmap

    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.set_stamp(2, np.full((1, 1), ref.firstgid, dtype=np.uint32))
    assert wmap.read_wmap(wmap.wmap_bytes(doc)).stamps[2] is not None

    # Past the end of every tileset this map has.
    doc.set_stamp(2, np.full((1, 1), ref.last_gid + 9, dtype=np.uint32))
    with pytest.raises(ValueError, match="stamp 2"):
        wmap.read_wmap(wmap.wmap_bytes(doc))


def test_the_sheet_is_off_until_a_tileset_is_chosen():
    from warlock.studio.panes import plotter_tileset_editor

    doc = _doc()
    tab = plotter_state.PlotterDoc(doc=doc, title="m")
    state = plotter_state.PlotterState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(plotter=state))
    assert plotter_tileset_editor.active(ctx) is False
    doc.add_tileset(_tileset())
    state.editing_tileset = 0
    assert plotter_tileset_editor.active(ctx) is True
    # An index the list cannot honour is not a sheet: the map stays on screen.
    state.editing_tileset = 7
    assert plotter_tileset_editor.active(ctx) is False


def test_the_editor_offers_no_reordering():
    """Order *is* firstgid order, baked into every painted cell, so reordering
    means renumbering the map. Tiled reorders its tabs, not its ids."""

    import inspect

    from warlock.studio.panes import plotter_tileset_editor

    source = inspect.getsource(plotter_tileset_editor)
    assert "move_tileset" not in source
    assert "reorder" not in source.lower().split('"""')[2]


# --- one gesture, one undo step (the 2026-09-07 audit, plotter-01) ----------


def test_tileset_editor_tile_class_and_duration_and_wang_name_typing_is_one_undo_step():
    """``set_tile_meta``/``set_wang_colour`` push an unconditional
    ``history.push``, so a field drawn with no ``fold_undo`` between it and the
    write pushes one undo step per keystroke instead of one per gesture.

    Three fields had that shape: the per-tile Class field and the animation
    frame's duration in ``_tiles_tab``/``_animation_tab``, and the Wang colour
    Name field in ``_wang_colours`` -- which already folded its hue-bar drag
    and its Probability field but not the Name text box beside them. The
    palette's own copy of Class/Probability (``plotter_tileset.py``) carried a
    comment claiming this editor's copy "already does" fold, which was false
    until this fix. Positional, like the popup-door scan in
    ``tests/test_undo_gesture_doors.py``: draw, fold, act.
    """
    import inspect

    from warlock.studio.panes import plotter_tileset_editor as editor

    # Each check is bounded to the gap between one field and the *next* one
    # drawn (or the shared write, for the last field in a group) -- not merely
    # "a fold exists somewhere before the write", which a neighbour's own fold
    # would satisfy for free and prove nothing about the field in question.
    tiles_source = inspect.getsource(editor._tiles_tab)
    after_class = tiles_source.split('"##ts-class"', 1)[1]
    before_probability = after_class.split('"Probability"', 1)[0]
    assert "controls.fold_undo(" in before_probability, (
        "Class field is not folded before the Probability field is drawn"
    )
    after_probability = tiles_source.split('"Probability"', 1)[1]
    before_write = after_probability.split("tab.doc.set_tile_meta(", 1)[0]
    assert "controls.fold_undo(" in before_write, (
        "Probability field writes before it is folded"
    )

    animation_source = inspect.getsource(editor._animation_tab)
    after_duration = animation_source.split('"ms"', 1)[1]
    before_write = after_duration.split("write(edited)", 1)[0]
    assert "controls.fold_undo(" in before_write, (
        "frame duration writes before it is folded"
    )

    wang_source = inspect.getsource(editor._wang_colours)
    after_name = wang_source.split('"##name"', 1)[1]
    before_swatch = after_name.split("controls.color_edit4(", 1)[0]
    assert "controls.fold_undo(" in before_swatch, (
        "Wang colour Name field is not folded before the swatch is drawn"
    )


# --- the picker does not draw one button per tile (the 2026-09-11 audit,
# finding plotter-04) --------------------------------------------------------


class _HugeTileset:
    """A stand-in whose only trait ``_tile_grid`` reads is its length.

    ``len()`` looks ``__len__`` up on the *type*, not the instance, so a
    ``SimpleNamespace`` carrying it as an attribute would not answer to
    ``len()`` -- hence a real (tiny) class rather than the fixtures this file
    otherwise builds real ``Tileset`` pixel buffers for.
    """

    def __len__(self) -> int:
        return 20000


def test_the_tiles_tab_does_not_draw_a_button_per_tile_on_a_large_tileset(ui, monkeypatch):
    """``_tile_grid`` used to submit one ``controls.button`` per tile in the
    tileset, every frame the tab was open, with nothing bounding the tileset's
    tile *count* -- unlike the sibling picker (``plotter_tileset.py``), whose
    own docstring gives the reason a picker does not do this: "a 16x16
    tileset is 256 buttons, and imgui would spend a per-item id, a hover test
    and a draw call on each of them every frame." A user-imported sheet
    sliced small (a 2048x2048 PNG at 16x16 is 16,384 tiles) made this tab draw
    over sixteen thousand buttons on the frame thread -- two orders of
    magnitude past what the sibling picker was built to avoid.

    A real headless imgui window, sized far smaller than a 20,000-tile grid,
    proves the fix by counting how many buttons actually got submitted: the
    unfixed code submits one per tile regardless of the window; the fixed
    ``ImGuiListClipper`` submits only the rows the visible, scrolled region
    can show.
    """
    from warlock.studio.panes import plotter_tileset_editor as editor

    calls: list[int] = []
    monkeypatch.setattr(
        editor.controls, "button", lambda *a, **k: (calls.append(1), False)[1]
    )
    state = plotter_state.PlotterState()
    ref = SimpleNamespace(tileset=_HugeTileset())

    ui.new_frame()
    ui.begin("##host")
    ui.begin_child("##scroll", (380.0, 260.0))
    editor._tile_grid(SimpleNamespace(), state, ref)
    ui.end_child()
    ui.end()
    ui.end_frame()

    # A small scrolled window can show, at most, a couple of dozen 48px
    # tiles -- nowhere near the 20,000 in the tileset. The bound is generous
    # on purpose: it only has to separate "clipped to the visible region"
    # from "one button per tile", not pin the exact row count.
    assert 0 < len(calls) < 1000, (
        f"expected a clipped, bounded number of buttons; drew {len(calls)}"
    )
