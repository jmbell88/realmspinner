"""Regression tests closing the 2026-10-14 audit's Packwright findings handled by F39
(packwright-03, -04, -05, -06, -07, -11, -13).

Each test's name is the claim.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import rpack

from .test_packwright_mode import FakeCtx, _Done, _tab
from .test_packwright_tileset_mode import _dup_sheet, _park


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- packwright-03: an Inker document's sprites are keyed on the document -----


def _inker_doc(rgb: tuple[int, int, int]) -> Any:
    from realmspinner.kernels.pixel.document import Document

    doc = Document.blank(16, 16)
    layer = doc.stack[0]
    pixels = np.array(layer.pixels).copy()
    pixels[4:12, 4:12] = (*rgb, 255)
    layer.pixels = pixels
    return doc


def test_two_inker_documents_with_one_title_stay_two_sets_of_sprites() -> None:
    """Both documents are "Untitled" (every unsaved one is), so keying on the
    title stem made the second "Add to Packwright" replace the first's art."""
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    first = SimpleNamespace(title="Untitled", uid="pd1", doc=_inker_doc((255, 0, 0)))
    second = SimpleNamespace(title="Untitled", uid="pd2", doc=_inker_doc((0, 0, 255)))

    packwright_mode.add_inker_document(ctx, first)
    packwright_mode.add_inker_document(ctx, second)

    sprites = tab.doc.sprites()
    assert len(sprites) == 2, [s.key for s in sprites]
    assert len({s.key for s in sprites}) == 2
    assert {tuple(s.pixels[8, 8][:3]) for s in sprites} == {(255, 0, 0), (0, 0, 255)}
    # The display name still comes from the title, not from the identity.
    assert all("pd" not in s.name and "#" not in s.name for s in sprites), [
        s.name for s in sprites
    ]


def test_re_adding_one_inker_document_after_a_rename_picks_up_the_edit() -> None:
    """Identity is the document, so a retitle does not make duplicates."""
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    inker_tab = SimpleNamespace(title="Untitled", uid="pd7", doc=_inker_doc((255, 0, 0)))
    packwright_mode.add_inker_document(ctx, inker_tab)
    inker_tab.title = "hero.ora"
    packwright_mode.add_inker_document(ctx, inker_tab)
    assert len(tab.doc.sprites()) == 1


# --- packwright-04: a tile-set key is not the sheet's absolute path ----------


def test_a_tileset_import_does_not_write_the_sheets_absolute_path_into_the_rpack() -> None:
    sheet_path = Path(r"C:\Users\someartist\Secret Project\tiles.png")
    sheet = np.full((16, 16, 4), 255, np.uint8)
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": (str(sheet_path), sheet_path.stem, sheet), "uid": tab.uid},
        ),
    )
    packwright_mode.ensure(ctx).tileset_cell = (8, 8)
    assert packwright_mode.import_tileset(ctx) is True
    assert tab.doc.sources

    with zipfile.ZipFile(io.BytesIO(rpack.rpack_bytes(tab.doc))) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        text = "".join(zf.read(n).decode("utf-8", "replace") for n in names)
    assert "someartist" not in text and "Secret Project" not in text
    assert all(s.key.startswith("tiles#") for s in tab.doc.sources), [
        s.key for s in tab.doc.sources
    ]


def test_two_different_sheets_with_one_name_still_key_apart() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    sheet = np.full((8, 8, 4), 255, np.uint8)
    for folder in ("a", "b"):
        state.tileset_import = (str(Path("C:/x") / folder / "tiles.png"), "tiles", sheet)
        state.tileset_import_uid = tab.uid
        state.tileset_cell = (8, 8)
        assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 2


# --- packwright-05: Import does no slicing or dedup on the frame thread ------


def test_import_tileset_does_no_slicing_or_dedup_on_the_frame_thread(monkeypatch) -> None:
    from realmspinner.studio.modes.packwright.engine import sources as engine_sources

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = _dup_sheet()
    _park(ctx, tab, pixels)
    state.tileset_cell = (4, 4)
    state.tileset_dedup = True
    state.tileset_dedup_flips = True

    # The popup's own off-thread task computes the sprites; Import then only
    # adds them.
    packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)
    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("sliced or deduped on the frame thread")

    monkeypatch.setattr(engine_sources, "sprites_from_tileset", boom)
    monkeypatch.setattr(engine_sources, "dedup_tiles", boom)

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 2


def test_the_slice_preview_grid_comes_from_the_preview_task_not_the_frame(monkeypatch) -> None:
    """``_occupancy_for`` ran ``tileset_occupancy`` on the frame thread on every
    tile-size edit (60-130 ms on an 8192 px sheet)."""
    from realmspinner.studio.modes.packwright.engine import sources as engine_sources
    from realmspinner.studio.modes.packwright.ui.panes import sources as packwright_sources

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = _dup_sheet()
    _park(ctx, tab, pixels)
    state.tileset_cell = (4, 4)
    packwright_sources._slice_grid_cache = None

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("occupancy computed on the frame thread")

    monkeypatch.setattr(engine_sources, "tileset_occupancy", boom)
    assert packwright_sources._occupancy_for(pixels, (4, 4), compute=False) is None

    monkeypatch.undo()
    packwright_mode.request_tileset_preview(ctx, state, "sheet.png", "sheet", pixels)
    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    monkeypatch.setattr(engine_sources, "tileset_occupancy", boom)

    grid = packwright_sources._occupancy_for(pixels, (4, 4), compute=False)
    assert grid is not None and grid.shape == (2, 2)
    packwright_sources._slice_grid_cache = None


# --- packwright-06: a drop stops decoding at the pack's pixel budget ---------


def _write_pngs(folder: Path, count: int, side: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = folder / f"t{index:02d}.png"
        arr = np.zeros((side, side, 4), np.uint8)
        arr[..., 0] = index + 1
        arr[..., 3] = 255
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def test_a_drop_stops_decoding_once_the_packs_pixel_budget_is_spent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(rpack, "MAX_DOCUMENT_PIXELS", 1000)
    paths = _write_pngs(tmp_path, 4, 20)  # 400 px each: two fit, the third does not
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)

    packwright_mode.add_source_paths(ctx, paths)

    decoded = ctx.result["sprites"]
    assert len(decoded) == 2, "the rest must not be decoded at all"
    packwright_mode.on_task_done(ctx, _Done(f"packwright-add:{tab.uid}:1", ctx.result))
    assert len(tab.doc.sources) == 2
    assert any("Stopped after 2 of 4" in message for message, _ in ctx.toasts), ctx.toasts


def test_a_drop_refuses_a_file_past_the_source_ceiling_before_decoding_it(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(rpack, "MAX_SOURCE_PIXELS", 100)
    paths = _write_pngs(tmp_path, 2, 20)
    converted: list[int] = []
    real = Image.Image.convert

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        converted.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", spy)
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)

    packwright_mode.add_source_paths(ctx, paths)

    assert converted == [], "an oversize file must be refused from its header"
    assert ctx.result["sprites"] == []
    packwright_mode.on_task_done(ctx, _Done(f"packwright-add:{tab.uid}:1", ctx.result))
    assert not tab.doc.sources
    assert any("Stopped after 0 of 2" in message for message, _ in ctx.toasts), ctx.toasts


def test_a_redrop_of_files_already_in_a_full_pack_is_not_stopped(tmp_path, monkeypatch) -> None:
    """A re-add replaces rather than grows, so its pixels are credited."""
    paths = _write_pngs(tmp_path, 2, 20)
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    packwright_mode.add_source_paths(ctx, paths)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-add:{tab.uid}:1", ctx.result))
    assert len(tab.doc.sources) == 2
    monkeypatch.setattr(rpack, "MAX_DOCUMENT_PIXELS", 800)  # exactly full

    packwright_mode.add_source_paths(ctx, paths)
    assert len(ctx.result["sprites"]) == 2


# --- packwright-07: a failed first pack is named, not "nothing packed" -------


def test_packed_why_reports_a_failed_first_pack_not_nothing_packed() -> None:
    from realmspinner.studio.modes.packwright.engine.sources import Sprite
    from realmspinner.studio.modes.packwright.ui.panes import bridge

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    pixels = np.zeros((600, 600, 4), np.uint8)
    pixels[..., 3] = 255
    tab.doc.add_source(Sprite(key="big", name="big", pixels=pixels))
    packwright_mode.set_settings(ctx, tab, mode="maxrects", max_size=256)
    try:
        packwright_mode.request_pack(ctx, tab)
    except Exception as exc:  # FakeCtx runs the body inline; the runner would fail the task
        packwright_mode.on_task_failed(
            ctx, _Done(f"packwright-pack:{tab.uid}", message=str(exc))
        )
    assert tab.layout is None and tab.pack_error

    why = bridge._packed_why(tab)
    assert "Nothing is packed yet" not in why
    assert why == tab.pack_error


# --- packwright-11: the remedy lists only what would change something --------


def test_pack_error_remedy_omits_trim_when_trim_is_already_on() -> None:
    from realmspinner.studio.modes.packwright.ui.panes import items

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="maxrects", trim=True)
    assert "trim" not in items._pack_error_remedy(tab).lower()
    packwright_mode.set_settings(ctx, tab, trim=False)
    assert "trim" in items._pack_error_remedy(tab).lower()


def test_pack_error_remedy_omits_raising_the_max_size_at_the_ceiling() -> None:
    from realmspinner.kernels.sheet import MAX_ATLAS_PX
    from realmspinner.studio.modes.packwright.ui.panes import items

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, max_size=MAX_ATLAS_PX)
    remedy = items._pack_error_remedy(tab)
    assert "max size" not in remedy.lower()
    assert "split" in remedy.lower()


# --- packwright-13: the Max size combo lists the document's own value --------


def test_max_size_combo_lists_a_non_standard_document_value() -> None:
    from realmspinner.studio.modes.packwright.ui.panes import settings

    keys = [key for key, _label in settings.max_size_options(1500)]
    assert "1500" in keys
    assert keys == sorted(keys, key=int)
    standard = [key for key, _label in settings.max_size_options(1024)]
    assert standard == [str(s) for s in settings.SIZES]
