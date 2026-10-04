"""Regression tests closing the 2026-10-03 audit's Medium Packwright findings
(batch packwright-1: packwright-01 .. packwright-08).

Each test's name is the claim, and each failed against the unfixed code before
the corresponding fix landed.
"""

from __future__ import annotations

import io
import json
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import rpack
from realmspinner.studio.modes.packwright.engine.document import PackDoc, Source
from realmspinner.studio.modes.packwright.engine.sources import (
    Sprite,
    SpriteMeta,
)
from realmspinner.studio.modes.packwright.ui.panes import sources as packwright_sources

from .test_packwright_mode import FakeCtx, _edited, _pack, _sprite, _tab
from .test_packwright_tileset_mode import _park, _sheet
from .test_rpack import _doc, _rewrite_member


@pytest.fixture(autouse=True)
def _no_slice_cache_leak():
    packwright_sources._slice_grid_cache = None
    yield
    packwright_sources._slice_grid_cache = None


class _Tex:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


# --- packwright-01 ------------------------------------------------------------


def test_a_successful_tileset_import_releases_the_slice_preview_texture_and_grid():
    """The Import button resets the parked-sheet fields itself, so the closed-
    popup branch's "something is parked" guard is already False on the next
    frame and nothing forgot the slice preview's GL texture or its cached
    occupancy grid."""
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_cell = (4, 4)
    texture = _Tex()
    key = f"{packwright_sources._SLICE_TEX_PREFIX}{id(state.tileset_import[2])}"
    ctx.state.preview[key] = texture
    packwright_sources._slice_grid_cache = ((1, (4, 4)), np.ones((2, 2), dtype=bool))

    assert packwright_mode.import_tileset(ctx) is True

    assert texture.released, "the slice preview's texture outlived the import"
    assert key not in ctx.state.preview
    assert packwright_sources._slice_grid_cache is None


# --- packwright-02 ------------------------------------------------------------


def test_leaving_packwright_with_the_tileset_popup_open_does_not_leave_modal_open_true():
    """``tileset_popup_open`` answers from a flag only Packwright's own pane
    clears, so leaving the mode with the popup up kept ``modal_open`` True in
    every other mode and ``events._events`` dropped every KEYDOWN."""
    from realmspinner.studio import dialogs
    from realmspinner.studio.shell import events

    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_import_open = True
    assert packwright_sources.tileset_popup_open(ctx) is True

    ctx.confirms = SimpleNamespace(pending=None)
    ctx.prompts = SimpleNamespace(pending=None)
    ctx.state.mode = "inker"
    events._leave_mode_if_needed(ctx, "packwright")

    assert packwright_sources.tileset_popup_open(ctx) is False
    assert state.tileset_import is None, "the parked sheet was held after leaving"
    assert dialogs.modal_open(ctx) is False


# --- packwright-03 ------------------------------------------------------------


def test_adding_many_sprites_in_one_gesture_is_one_undo_step():
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    sprites = [_sprite(f"f{i}") for i in range(6)]

    packwright_mode._add_sprites(ctx, tab, sprites)
    assert len(tab.doc.sources) == 6

    tab.doc.undo()
    assert tab.doc.sources == [], "one Ctrl+Z must undo the whole add"
    tab.doc.redo()
    assert len(tab.doc.sources) == 6


# --- packwright-04 ------------------------------------------------------------


def test_the_packwright_journal_encode_does_no_png_work_on_the_frame_thread(
    tmp_path, monkeypatch
):
    from realmspinner.studio import journal

    ctx = FakeCtx()
    tab = _tab(ctx, sources=3)
    ctx.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path))
    tab.doc.set_settings(padding=4, extrude=2)  # dirty, so the journal would copy it
    calls: list[str] = []
    real = rpack.png_bytes

    def counting(pixels):
        calls.append(threading.current_thread().name)
        return real(pixels)

    monkeypatch.setattr(rpack, "png_bytes", counting)
    deferred: list = []
    ctx.submit = lambda key, run, *a: deferred.append(run) or True

    assert journal.write(ctx, packwright_mode.JOURNAL, tab) is True

    assert calls == [], "journal.write ran the PNG encode before handing off to a task"
    assert len(deferred) == 1
    deferred[0]()  # the task's half
    assert len(calls) == 3
    pair = [p for p in tmp_path.iterdir() if p.name.endswith(".rpack")]
    assert len(pair) == 1
    assert len(rpack.read_rpack(pair[0].read_bytes()).sources) == 3


# --- packwright-05 ------------------------------------------------------------


def _row_sheet(tiles: int) -> np.ndarray:
    """One row of ``tiles`` 4 px cells, every one distinct and occupied."""
    pixels = np.zeros((4, tiles * 4, 4), dtype=np.uint8)
    for i in range(tiles):
        pixels[1, i * 4 + 1] = (i % 251, (i // 251) % 251, 7, 255)
    return pixels


def _key_reads_for_import(monkeypatch, tiles: int) -> int:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _row_sheet(tiles))
    state.tileset_cell = (4, 4)
    reads = [0]
    real = Source.key.fget

    def counting(self):
        reads[0] += 1
        return real(self)

    monkeypatch.setattr(Source, "key", property(counting))
    assert packwright_mode.import_tileset(ctx) is True
    monkeypatch.setattr(Source, "key", property(real))
    assert len(tab.doc.sources) == tiles
    return reads[0]


def test_importing_a_ceiling_sized_tileset_does_not_scale_quadratically_with_the_tile_count(
    monkeypatch,
):
    """Each sprite used to pay a scan of every source already held (the
    uniqueness lookup, ``has_key`` and ``total_pixels``): quadratic in the
    batch. Counted rather than timed -- linear is 4x for 4x the tiles,
    quadratic 16x."""
    small = _key_reads_for_import(monkeypatch, 256)
    large = _key_reads_for_import(monkeypatch, 1024)
    assert large < small * 6, (small, large)


# --- packwright-06 ------------------------------------------------------------


def test_replace_source_keeps_a_pivot_the_incoming_sprite_does_not_carry():
    doc = PackDoc()
    source = doc.add_source(_sprite("a"))
    doc.set_pivot(source.uid, (1.0, 2.0))

    doc.replace_source(source.uid, _edited("a"))  # a loose PNG: empty meta

    held = doc.source(source.uid).sprite
    assert held.meta.pivot == (1.0, 2.0), "re-adding an edited PNG wiped its pivot"
    assert np.array_equal(held.pixels, _edited("a").pixels), "the new pixels still land"


def test_replace_source_picks_up_changed_producer_metadata_with_identical_pixels():
    doc = PackDoc()
    first = _sprite("a")
    source = doc.add_source(
        Sprite(key="a", name="a", pixels=first.pixels, meta=SpriteMeta(pivot=(1.0, 1.0)))
    )
    head = doc.history.head

    doc.replace_source(
        source.uid,
        Sprite(key="a", name="a", pixels=first.pixels, meta=SpriteMeta(pivot=(3.0, 2.0))),
    )

    assert doc.history.head != head, "an Inker pivot edit was dropped as 'unchanged'"
    assert doc.source(source.uid).sprite.meta.pivot == (3.0, 2.0)


# --- packwright-07 ------------------------------------------------------------


def test_read_rpack_refuses_a_truncated_png_member_by_name():
    doc = _doc()
    with zipfile.ZipFile(io.BytesIO(rpack.rpack_bytes(doc))) as zf:
        whole = zf.read("sources/1.png")
    data = _rewrite_member(doc, "sources/1.png", whole[: len(whole) * 6 // 10])

    with pytest.raises(ValueError, match="sources/1.png"):
        rpack.read_rpack(data)


# --- packwright-08 ------------------------------------------------------------


def test_export_refuses_to_overwrite_an_existing_json_the_picker_never_confirmed(
    tmp_path, monkeypatch
):
    from realmspinner.service.errors import Invalid
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    (tmp_path / "hero.json").write_bytes(b'{"mine": true}')
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "hero.png")

    with pytest.raises(Invalid, match=r"hero\.json"):
        packwright_mode.export_files(ctx, tab)

    assert (tmp_path / "hero.json").read_bytes() == b'{"mine": true}'
    assert not (tmp_path / "hero.png").exists()


def test_re_exporting_over_its_own_earlier_sidecar_is_still_allowed(tmp_path, monkeypatch):
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "hero.png")
    packwright_mode.export_files(ctx, tab)
    packwright_mode.export_files(ctx, tab)  # the same atlas again: no refusal

    assert json.loads((tmp_path / "hero.json").read_text(encoding="utf-8"))["meta"]


def test_save_as_refuses_to_overwrite_an_rpack_whose_name_the_picker_never_confirmed(
    tmp_path, monkeypatch
):
    from realmspinner.service.errors import Invalid
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    (tmp_path / "atlas.rpack").write_bytes(b"the user's other atlas")
    # Typed without the suffix: the picker confirmed ``atlas``, not ``atlas.rpack``.
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "atlas")

    with pytest.raises(Invalid, match=r"atlas\.rpack"):
        packwright_mode.save_as(ctx, tab)

    assert (tmp_path / "atlas.rpack").read_bytes() == b"the user's other atlas"
