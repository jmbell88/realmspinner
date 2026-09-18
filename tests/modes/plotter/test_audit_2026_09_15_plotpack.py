"""Regression tests for the 2026-09-15 audit's Plotter/Packwright findings.

- plotter-04 (`src/warlock/studio/modes/plotter/tilesets.py`): ``land_tileset``
  applied a sheet's recorded projection onto the map unconditionally once
  ``use_as_tileset`` had decided, at *submit* time, that the map was
  unpainted. The decode is a task round trip; a paint stroke landing in that
  window reached ``land_tileset`` with a map that was no longer empty, and the
  stale answer silently reprojected it. Fixed by re-checking paint state at
  landing and falling back to the same parked ``SheetLattice`` confirmation a
  painted map gets at submit time.
- plotter-05 (`src/warlock/studio/modes/plotter/engine/_map_layers.py`): the source-only
  branch of ``set_image_pixels`` returned ``False`` ("nothing changed") even
  when ``set_layer_props`` had just pushed a real undo step for the changed
  source, contradicting the function's own "one compound step" docstring for
  a caller that gates a toast or a re-render on the return value.
- packwright-02 (`src/warlock/studio/modes/packwright/mode.py`): the second-landing
  refusal for a tile-sheet import checked ``tileset_import_open``, which only
  the pane's own draw sets. Two landings inside one poll batch -- both
  processed before a frame is ever drawn -- both saw it ``False`` and the
  second silently swapped the first's parked pixels.

plotter-01, plotter-02 and plotter-03 (the three unfolded undo doors) extend
``test_undo_gesture_doors.py``'s existing parametrized list instead of adding
new tests here. packwright-01 (`atomic.staged_set`'s cleanup-list ordering)
is a new test in ``test_atomic_writes.py``, the file that already owns
``staged_set``'s other regression tests.
"""

from __future__ import annotations

import numpy as np

from warlock.kernels.grid2d import gid as gidlib
from warlock.kernels.grid2d.tileset import Tileset
from warlock.studio.modes.plotter import mode as plotter_mode
from warlock.studio.modes.plotter import tilesets as plotter_tilesets
from warlock.studio.modes.plotter.engine import project

# --- plotter-04 -----------------------------------------------------------


def test_use_as_tileset_reprojection_rechecks_paint_state_at_landing_not_at_request():
    """Against the unfixed ``land_tileset``, this map's projection silently
    moves from orthogonal to isometric even though a tile was painted onto it
    before landing -- the exact hazard :class:`SheetLattice` exists to ask
    the user about on a painted map, just arriving one task round trip late.
    """
    from .test_plotter_mode import FakeCtx

    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), projection=project.ORTHOGONAL)
    state = plotter_mode.ensure(ctx)

    # The paint stroke that lands in the window ``use_as_tileset``'s
    # submit-time ``unpainted`` read cannot see: a task round trip separates
    # the read from this landing, and nothing here reruns it.
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 0, 0, np.array([[1]], gidlib.DTYPE))

    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    tileset = Tileset(name="iso-sheet", pixels=pixels, tile_w=16, tile_h=16)
    result = {
        "tileset": tileset,
        "source": "",
        "uid": tab.uid,
        "projection": project.ISOMETRIC,
    }
    before = tab.doc.projection

    plotter_tilesets.land_tileset(ctx, state, tab, result)

    assert tab.doc.projection == before, "a painted map's lattice must not move under it"
    assert tileset not in [ref.tileset for ref in tab.doc.tilesets], (
        "the tileset must not land silently either -- it is parked with the conflict"
    )
    assert state.sheet_import is not None, "the conflict must be parked for the user"
    parked_uid, _name, _source, _pixels, grid = state.sheet_import
    assert parked_uid == tab.uid
    assert isinstance(grid, plotter_tilesets.SheetLattice)
    assert grid.view == project.ISOMETRIC
    assert grid.lattice == before


def test_use_as_tileset_reprojection_still_applies_on_a_map_that_stayed_empty():
    """The re-check must not turn into a refusal for the ordinary case: an
    empty map that is still empty at landing takes the sheet's lattice
    exactly as it did before this fix, in one step.
    """
    from .test_plotter_mode import FakeCtx

    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), projection=project.ORTHOGONAL)
    state = plotter_mode.ensure(ctx)

    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    tileset = Tileset(name="iso-sheet", pixels=pixels, tile_w=16, tile_h=16)
    result = {
        "tileset": tileset,
        "source": "",
        "uid": tab.uid,
        "projection": project.ISOMETRIC,
    }

    plotter_tilesets.land_tileset(ctx, state, tab, result)

    assert tab.doc.projection == project.ISOMETRIC
    assert tileset in [ref.tileset for ref in tab.doc.tilesets]
    assert state.sheet_import is None


# --- plotter-05 -------------------------------------------------------------


def _picture(size: int = 4) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = 120
    out[..., 3] = 255
    return out


def test_set_image_pixels_reports_true_when_only_the_source_changes():
    """Against the unfixed method this returns ``False`` even though
    ``set_layer_props`` just pushed a real ``LayerPropsEdit`` for the new
    source -- a caller reading the return value as "nothing changed" while
    an undo step landed behind its back."""
    from warlock.studio.modes.plotter.engine.tilemap import MapDoc

    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Tiles")
    layer = doc.add_image_layer("Picture")
    picture = _picture()
    doc.set_image_pixels(layer.uid, picture, source="a.png")
    before = len(doc.history)

    changed = doc.set_image_pixels(layer.uid, picture, source="b.png")

    assert changed is True, "a step landed (the source changed) and must be reported"
    assert len(doc.history) == before + 1
    assert doc.layer(layer.uid).source == "b.png"


def test_set_image_pixels_still_reports_false_when_truly_nothing_changed():
    """The sibling case, unaffected by the fix: identical pixels and an
    identical source push nothing and must go on reporting ``False``."""
    from warlock.studio.modes.plotter.engine.tilemap import MapDoc

    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Tiles")
    layer = doc.add_image_layer("Picture")
    picture = _picture()
    doc.set_image_pixels(layer.uid, picture, source="a.png")
    before = len(doc.history)

    changed = doc.set_image_pixels(layer.uid, picture, source="a.png")

    assert changed is False
    assert len(doc.history) == before


# --- packwright-02 -----------------------------------------------------------


def test_two_tileset_landings_in_the_same_poll_batch_do_not_silently_swap_the_parked_import():
    """Against the unfixed handler this passes ``tileset_import_open is
    False`` in both cases -- the shape two results processed in one poll
    batch take, since neither one has been drawn yet -- and the second
    landing silently overwrites the first's parked pixels with no toast.
    """
    from modes.packwright.test_packwright_mode import FakeCtx, _Done
    from warlock.studio.modes.packwright import mode as packwright_mode

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)

    first = np.zeros((4, 4, 4), dtype=np.uint8)
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("first.png", "first", first), "uid": tab.uid},
        ),
    )
    # No frame has been drawn between the two landings, so the pane's own
    # "open once a new import lands" check never ran and this is still its
    # default -- the exact "same poll batch" shape the finding names.
    assert state.tileset_import_open is False

    second = np.ones((4, 4, 4), dtype=np.uint8)
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("second.png", "second", second), "uid": tab.uid},
        ),
    )

    assert state.tileset_import[1] == "first", "the parked import must not be swapped"
    assert ctx.toasts, "a dropped second import must say so"
