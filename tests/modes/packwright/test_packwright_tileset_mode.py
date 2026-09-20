"""The tile-sheet door on Packwright's controller.

The decode task parks the sheet on the state, the popup's answer turns it into
sprites through :func:`packwright_mode.import_tileset`, and the key carries the
tile size so a re-cut is not a duplicate. Fakes follow
``tests/modes/packwright/test_packwright_mode.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from realmspinner.studio.modes.packwright import mode as packwright_mode


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, *, accept: bool = True) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _Done:
    def __init__(self, key: str, result: Any = None, message: str = "") -> None:
        self.key = key
        self.result = result
        self.message = message


def _sheet() -> np.ndarray:
    """2 x 2 cells of 4 px; the bottom-right cell is empty."""
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    for row, column in ((0, 0), (0, 1), (1, 0)):
        pixels[row * 4 + 1, column * 4 + 1] = (9, 9, 9, 255)
    return pixels


def _park(ctx: FakeCtx, tab: Any, pixels: np.ndarray) -> None:
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("sheet.png", "sheet", pixels), "uid": tab.uid},
        ),
    )


def test_the_ask_submits_under_its_own_prefix() -> None:
    ctx = FakeCtx(accept=False)  # never run the dialog-opening task in a test
    tab = packwright_mode.new_document(ctx)
    packwright_mode.ask_add_tileset(ctx)
    assert ctx.submitted == [f"packwright-tileset:{tab.uid}"]


def test_the_decode_landing_parks_the_sheet_on_the_state() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    _park(ctx, tab, _sheet())
    state = packwright_mode.ensure(ctx)
    assert state.tileset_import is not None
    assert state.tileset_import[1] == "sheet"
    assert state.tileset_import_open is False


def test_a_landing_for_a_closed_tab_is_dropped() -> None:
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    packwright_mode.on_task_done(
        ctx, _Done("packwright-tileset:gone", {"tileset": ("p", "p", _sheet()), "uid": "gone"})
    )
    assert packwright_mode.ensure(ctx).tileset_import is None


def test_import_adds_only_occupied_cells_and_rearms_the_pack() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    tab.pack_dirty = False
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_cell = (4, 4)

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 3
    assert tab.pack_dirty is True
    assert state.tileset_import is None
    assert ctx.toasts[-1][0] == "Added 3 tile(s)."


def test_a_recut_at_another_tile_size_is_not_a_duplicate() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_cell = (4, 4)
    packwright_mode.import_tileset(ctx)
    before = len(tab.doc.sources)

    _park(ctx, tab, _sheet())
    state.tileset_cell = (2, 2)
    packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) > before


def test_the_same_cut_twice_is_a_duplicate_and_says_so() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    for _ in range(2):
        _park(ctx, tab, _sheet())
        state.tileset_cell = (4, 4)
        packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) == 3
    assert ctx.toasts[-1][0] == "Those tiles are already in this atlas, unchanged."


def test_import_with_nothing_parked_is_a_no_op() -> None:
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    assert packwright_mode.import_tileset(ctx) is False


# --- packwright-02 (2026-09-11 audit): the landing must not follow focus -----


def test_a_tileset_import_lands_in_the_tab_that_requested_it_not_whichever_is_active() -> None:
    """Tab A asks to add a tile set -- the decode is submitted under a key
    carrying A's uid. Before it lands, the frame loop keeps pumping and the
    user switches to tab B, ordinary since a background task never blocks it.
    The decode lands (``on_task_done``, naming A in its key); only then does
    the user answer the tile-size popup, which is drawn over whichever tab is
    now active -- B. The sheet must still land in A, the tab that asked, not
    B, the tab that merely happened to be focused when Import was pressed."""
    ctx = FakeCtx()
    tab_a = packwright_mode.new_document(ctx)
    tab_b = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)

    state.activate(tab_a.uid)
    key = f"packwright-tileset:{tab_a.uid}"

    # The user switches away from A while the decode is presumed in flight.
    state.activate(tab_b.uid)
    assert state.active is tab_b

    packwright_mode.on_task_done(
        ctx, _Done(key, {"tileset": ("sheet.png", "sheet", _sheet()), "uid": tab_a.uid})
    )
    assert state.active is tab_b, "a background landing must not move the user's focus"

    state.tileset_cell = (4, 4)
    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab_a.doc.sources) == 3, "the tab that asked for the tile set gets it"
    assert len(tab_b.doc.sources) == 0, "not the tab that happened to be active at Import"


def test_import_declines_with_a_toast_if_the_requesting_tab_has_since_closed() -> None:
    """The same window packwright-02 reproduces, one step later: the
    requesting tab can close between the sheet landing and the Import press.
    Silently adding the sheet to whichever tab is active would repeat the
    same bug by another route; silently doing nothing would look like Import
    did not work. It must say so."""
    ctx = FakeCtx()
    tab_a = packwright_mode.new_document(ctx)
    _park(ctx, tab_a, _sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_cell = (4, 4)

    state.close(tab_a.uid)  # the tab that asked is gone

    assert packwright_mode.import_tileset(ctx) is False
    assert state.tileset_import is None
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


# --- dedup at the import door (Part I) ----------------------------------------


def _dup_sheet(size: int = 4) -> np.ndarray:
    """Four cells: two identical, one their mirror, one genuinely different."""
    one = np.zeros((size, size, 4), dtype=np.uint8)
    one[..., :3] = (30, 30, 30)
    one[..., 3] = 255
    one[0, 0, :3] = (200, 40, 40)
    other = one.copy()
    other[size - 1, size - 1, :3] = (40, 200, 40)
    flipped = np.ascontiguousarray(one[:, ::-1])
    top = np.concatenate([one, one.copy()], axis=1)
    bottom = np.concatenate([flipped, other], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _park_dups(ctx: FakeCtx, pixels: np.ndarray, cell: int = 4) -> Any:
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    state.tileset_import = ("sheet.png", "sheet", pixels)
    # ``import_tileset`` now targets the tab named by ``tileset_import_uid``
    # (packwright-02, above) rather than whichever tab is active -- this
    # helper parks a sheet directly, bypassing the ``on_task_done`` landing
    # that would normally set it, so it has to set it too.
    state.tileset_import_uid = tab.uid
    state.tileset_cell = (cell, cell)
    return tab


def test_dedup_is_off_by_default_and_the_import_stays_byte_faithful():
    """A repack is a faithful repack unless somebody asks otherwise: an atlas
    quietly missing four tiles that happened to be rotations of each other is a
    bug report nobody can reproduce."""
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    assert state.tileset_dedup is False

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 4


def test_exact_duplicates_drop_when_the_flag_is_set():
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True

    assert packwright_mode.import_tileset(ctx) is True
    # The two identical cells become one; the mirror and the odd one survive.
    assert len(tab.doc.sources) == 3


def test_the_orientation_toggle_also_drops_the_mirror():
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True
    state.tileset_dedup_flips = True

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 2


def test_the_popup_preview_and_the_import_agree_on_the_dropped_count():
    """The popup-promise contract ``tileset_occupancy`` already enforces for
    emptiness, applied to the dedup the popup now also promises: both sides run
    the same ``dedup_tiles`` call over the same sprites."""
    from realmspinner.studio.modes.packwright.engine.sources import (
        dedup_tiles,
        sprites_from_tileset,
    )

    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True
    state.tileset_dedup_flips = True

    tile = state.tileset_cell
    promised, dropped = dedup_tiles(
        sprites_from_tileset(
            state.tileset_import[2],
            tile=tile,
            prefix=f"sheet.png@{tile[0]}x{tile[1]}",
            name="sheet",
        ),
        orientations=True,
    )
    packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) == len(promised)
    assert dropped == 2


# --- packwright-05: the popup's counts run off the frame thread ---------------


def test_the_tileset_preview_goes_through_ctx_submit_not_inline():
    """The 2026-09-07 audit's packwright-05: ``tileset_occupancy`` (and, with
    dedup on, ``dedup_tiles``/``sprites_from_tileset``) used to run
    synchronously on the frame thread every time the popup redrew with a
    changed key -- the module's own comment measures that at some hundreds of
    milliseconds on a full sheet. ``request_tileset_preview`` must submit a
    task, and the counts must not change until that task's result lands."""
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = _sheet()
    state.tileset_import = ("sheet.png", "sheet", pixels)
    state.tileset_cell = (4, 4)

    packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)

    assert ctx.submitted, "the counts must go through ctx.submit"
    assert state.tileset_preview_key is None, "not applied until the task lands"
    assert state.tileset_preview == (0, 0, 0, 0, 0)

    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    assert state.tileset_preview_key is not None
    _rows, _columns, kept, dropped, _duplicates = state.tileset_preview
    assert (kept, dropped) == (3, 1)


def test_a_clean_key_does_not_resubmit_every_frame():
    """The popup redraws every frame it is open; a memo key that never
    matches would submit a fresh task that often."""
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = _sheet()
    state.tileset_import = ("sheet.png", "sheet", pixels)
    state.tileset_cell = (4, 4)

    packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)
    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    ctx.submitted.clear()

    for _ in range(5):
        packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)
    assert ctx.submitted == []


def test_an_answer_for_a_superseded_cell_size_is_dropped_not_adopted():
    """The cell size can change again while a computation for the old size is
    still in flight -- typing a second digit before the first answer lands --
    and that stale answer must not overwrite the newer request's key."""
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = _sheet()
    state.tileset_import = ("sheet.png", "sheet", pixels)
    state.tileset_cell = (4, 4)

    packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)
    stale_key, stale_result = ctx.submitted[-1], ctx.result

    state.tileset_cell = (2, 2)  # the user kept typing before the answer landed
    packwright_mode.on_task_done(ctx, _Done(stale_key, stale_result))

    assert state.tileset_preview_key is None, "the stale answer must not land"
