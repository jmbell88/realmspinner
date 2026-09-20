"""Packwright's controller.

Two things carry this file. The save rules every editor here shares -- a failed
save clears the lock, and the head a save records is the one the encode wrote.
And one of its own: **the repack flag**, which is the ``findings_dirty`` lesson
applied a second time. ``TaskRunner.submit`` refuses a key already in flight and
nothing re-arms it, so a flag cleared regardless of whether the submit was
accepted silently drops every edit made while a pack was running.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import rpack
from realmspinner.studio.modes.packwright.engine.sources import Sprite


class FakeCtx:
    def __init__(self, svc: Any = None, *, accept: bool = True) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.cache = _Cache()
        self.viewer = None
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


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Done:
    def __init__(self, key: str, result: Any = None, message: str = "") -> None:
        self.key = key
        self.result = result
        self.message = message


def _sprite(key: str, w: int = 8, h: int = 6) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (200, 30, 30, 255)
    return Sprite(key=key, name=key, pixels=pixels)


def _edited(key: str) -> Sprite:
    """The same source after the user changed it on disk: one pixel differs.

    A sprite's pixels are read-only by construction (the frozen-source rule),
    so the edit is made on the array before it becomes one.
    """
    pixels = np.zeros((6, 8, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (200, 30, 30, 255)
    pixels[0, 0] = (1, 2, 3, 255)
    return Sprite(key=key, name=key, pixels=pixels)


def _tab(ctx: FakeCtx, *, sources: int = 3) -> Any:
    tab = packwright_mode.new_document(ctx)
    for index in range(sources):
        tab.doc.add_source(_sprite(f"s{index}"))
    tab.doc.mark_saved()
    tab.pack_dirty = True
    return tab


def _pack(ctx: FakeCtx, tab: Any) -> None:
    packwright_mode.request_pack(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-pack:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- packing ------------------------------------------------------------------


def test_a_pack_produces_a_layout_and_an_atlas():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    assert tab.layout is not None and tab.atlas is not None
    assert len(tab.layout.frames) == 3
    assert tab.atlas.shape[2] == 4
    assert not tab.packing and not tab.pack_dirty


def test_adopting_a_pack_bumps_the_generation_exactly_once():
    """The texture cache keys on it, so it has to move for a repack and not for
    anything else."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    first = tab.pack_generation
    tab.pack_dirty = True
    _pack(ctx, tab)
    assert tab.pack_generation == first + 1


def test_the_dirty_flag_is_cleared_only_when_the_submit_is_accepted():
    """The regression this flag exists for. The runner refuses a key already in
    flight; clearing regardless would drop the edit that arrived while the
    previous pack was running, permanently, because nothing re-arms it."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    packwright_mode.request_pack(ctx, tab)
    assert ctx.submitted == [f"packwright-pack:{tab.uid}"]
    assert tab.pack_dirty is True
    assert tab.packing is False


def test_a_clean_document_does_not_resubmit_every_frame():
    """``pump`` runs from the preview pane's draw, so it is called sixty times a
    second -- it has to be free when there is nothing to do."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    ctx.submitted.clear()
    for _ in range(10):
        packwright_mode.pump(ctx)
    assert ctx.submitted == []


def test_every_document_edit_re_arms_the_pack():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)

    packwright_mode.set_settings(ctx, tab, padding=6)
    assert tab.pack_dirty
    _pack(ctx, tab)

    packwright_mode.remove_source(ctx, tab.doc.sources[0].uid, tab)
    assert tab.pack_dirty


def test_the_source_index_is_cached_on_the_tab_and_rebuilt_only_on_repack():
    """packwright-07: the preview and items panes rebuilt a full ``key -> uid``
    index over every source on every single frame they drew, though
    ``pack_generation`` already exists as the memo key every consumer of it
    agrees on -- the index only has to describe whichever layout is actually
    on screen, and that only changes when a repack lands."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=3)
    _pack(ctx, tab)

    first = packwright_mode.source_index(tab)
    assert packwright_mode.source_index(tab) is first, "not rebuilt: no repack happened"

    tab.doc.add_source(_sprite("extra"))
    tab.pack_dirty = True
    _pack(ctx, tab)

    second = packwright_mode.source_index(tab)
    assert second is not first, "rebuilt: pack_generation moved"
    assert second["extra"] == tab.doc.sources[-1].uid


def test_a_pack_that_cannot_fit_carries_the_engines_remedy(monkeypatch):
    """``layout`` raises with the number *and* what to do about it, and only a
    ``ServiceError``'s text survives the task classifier -- so an unframed
    ``ValueError`` put "see the log for details" in the items pane instead."""
    from realmspinner.service.errors import ServiceError

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="maxrects", max_size=1)
    tab.pack_dirty = True
    with pytest.raises(ServiceError) as caught:
        packwright_mode.request_pack(ctx, tab)
    assert caught.value.message.startswith("That pack did not work")
    assert "raise the max size" in caught.value.message


def test_an_edit_made_during_a_pack_survives_its_adoption():
    """``adopt_pack`` must not touch ``pack_dirty``: the edit that arrived while
    the pack was in flight is not in the layout landing now, and the flag is
    cleared at the submit rather than at the adoption."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.request_pack(ctx, tab)
    result = ctx.result
    packwright_mode.set_settings(ctx, tab, padding=6)
    assert tab.pack_dirty

    packwright_mode.on_task_done(ctx, _Done(f"packwright-pack:{tab.uid}", result))
    assert tab.pack_dirty
    ctx.submitted.clear()
    packwright_mode.pump(ctx)
    assert ctx.submitted == [f"packwright-pack:{tab.uid}"]


def test_a_failed_pack_clears_packing_and_says_why():
    """An empty items list that looks like success is the worst outcome of a
    pack that could not fit."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.packing = True
    packwright_mode.on_task_failed(
        ctx, _Done(f"packwright-pack:{tab.uid}", message="does not fit")
    )
    assert not tab.packing
    assert tab.pack_error == "does not fit"


def test_an_empty_document_packs_to_nothing_rather_than_raising():
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    tab.pack_dirty = True
    packwright_mode.request_pack(ctx, tab)
    assert ctx.submitted == []
    assert tab.layout is None and not tab.pack_dirty


def test_impossible_settings_are_refused_with_the_reason():
    """``PackSettings`` validates on construction, so the pane never has to."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, padding=1, extrude=4)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "twice extrude" in ctx.toasts[-1][0]
    assert tab.doc.settings.extrude == 0


# --- sources ------------------------------------------------------------------


def test_a_batch_skips_what_is_already_there_rather_than_refusing():
    """Dropping twenty files of which one is already in the atlas should add
    nineteen. The document's own refusal stays the authority on what may
    coexist; this is the caller deciding what to ask for."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=2)
    added, replaced = packwright_mode._add_sprites(
        ctx, tab, [_sprite("s0"), _sprite("new")]
    )
    assert (added, replaced) == (1, 0), "the identical one is a skip, not a step"
    assert len(tab.doc.sources) == 3


def test_re_adding_a_changed_file_updates_the_sprite_it_is_already_holding():
    """``rpack``'s own contract -- "what the document records is what was
    packed; re-adding the source is how you pick up a change" -- which was
    false until 2026-09-03: a key already present was skipped whatever its
    pixels said, so the only way to pick up an edit was to delete the sprite
    first."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=2)
    uid = tab.doc.sources[0].uid
    edited = _edited("s0")

    added, replaced = packwright_mode._add_sprites(ctx, tab, [edited])

    assert (added, replaced) == (0, 1)
    assert len(tab.doc.sources) == 2, "a replacement, not a second sprite"
    assert tab.doc.source(uid) is not None, "and the same uid, so nothing else moved"
    assert tuple(tab.doc.source(uid).sprite.pixels[0, 0]) == (1, 2, 3, 255)
    assert tab.pack_dirty


def test_an_update_is_one_undo_step_back_to_the_old_picture():
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    uid = tab.doc.sources[0].uid
    before = tab.doc.source(uid).sprite.pixels.copy()
    packwright_mode._add_sprites(ctx, tab, [_edited("s0")])

    tab.doc.undo()

    assert np.array_equal(tab.doc.source(uid).sprite.pixels, before)


def test_a_sentence_names_both_halves_of_a_mixed_batch():
    """"Added 19" hides the twentieth file being the edited one the user
    dropped the folder for."""
    assert packwright_mode._added_sentence(19, 1) == "Added 19 sprite(s), updated 1."
    assert packwright_mode._added_sentence(0, 2) == "Updated 2 sprite(s)."
    assert packwright_mode._added_sentence(3, 0) == "Added 3 sprite(s)."
    assert "unchanged" in packwright_mode._added_sentence(0, 0)


def test_a_batch_add_that_trips_the_document_ceiling_refuses_instead_of_raising(monkeypatch):
    """The 2026-09-14 audit's packwright-01: ``PackDoc.add_source`` raises a
    bare ``ValueError`` at a ceiling (an oversized sprite, here, patched down
    so an ordinary test sprite trips it), and until this fix ``_add_sprites``
    let that propagate out of the loop -- past ``on_task_done``, into
    ``main.py``'s generic task-landing handler, which toasted "That did not
    finish landing: packwright-add:..." and left the sprite that landed
    *before* the trip with ``pack_dirty`` still unset."""
    from realmspinner.studio.modes.packwright.engine import rpack

    monkeypatch.setattr(rpack, "MAX_SOURCE_PIXELS", 10)
    ctx = FakeCtx()
    tab = _tab(ctx, sources=0)
    tab.pack_dirty = False
    small = _sprite("a", w=2, h=2)  # 4 pixels: under the patched ceiling
    big = _sprite("b", w=8, h=6)  # 48 pixels: over it, raises at the door
    never_reached = _sprite("c", w=2, h=2)

    added, replaced = packwright_mode._add_sprites(ctx, tab, [small, big, never_reached])

    assert (added, replaced) == (1, 0), "the batch stops at the sprite that tripped the ceiling"
    assert [s.key for s in tab.doc.sources] == ["a"], "c was never reached"
    assert tab.pack_dirty is True, "what landed before the ceiling is still dirty"
    assert ctx.toasts and ctx.toasts[-1][1] == "warn"
    assert "1 of 3" in ctx.toasts[-1][0]


def test_a_failed_repack_marks_the_atlas_it_left_on_screen():
    """A failed pack keeps the last good atlas -- a picture beats a blank pane
    -- but nothing said it was the old one, so the preview drew it unmarked and
    both exports wrote it."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    tab.adopt_pack(tab.doc.layout(), np.zeros((4, 4, 4), dtype=np.uint8))
    assert tab.pack_stale_why == ""

    packwright_mode.on_task_failed(
        ctx,
        type("Done", (), {"key": f"packwright-pack:{tab.uid}", "message": "too big"})(),
    )

    assert tab.atlas is not None, "the last good picture is kept"
    assert "too big" in tab.pack_stale_why


def test_neither_export_will_write_a_stale_atlas(monkeypatch):
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    tab.adopt_pack(tab.doc.layout(), np.zeros((4, 4, 4), dtype=np.uint8))
    tab.pack_error = "too big"
    monkeypatch.setattr(
        dialogs, "save_file", lambda *a, **k: pytest.fail("a picker opened")
    )

    packwright_mode.export_files(ctx, tab)
    packwright_mode.export_library(ctx, tab)

    assert ctx.submitted == [], "nothing was written"
    assert len(ctx.toasts) == 2
    assert all("atlas from before it" in message for message, _ in ctx.toasts)


def test_a_pack_that_lands_clears_the_mark():
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    tab.pack_error = "too big"
    tab.adopt_pack(tab.doc.layout(), np.zeros((4, 4, 4), dtype=np.uint8))
    assert tab.pack_stale_why == ""


def test_adding_with_nothing_open_says_so():
    ctx = FakeCtx()
    packwright_mode.ask_add_sources(ctx)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert ctx.submitted == []


def test_an_inker_document_contributes_one_sprite_per_frame():
    from realmspinner.kernels.pixel.document import Document

    ctx = FakeCtx()
    tab = _tab(ctx, sources=0)
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = (255, 0, 0, 255)
    doc.add_frame(copy=True)
    inker_tab = type("T", (), {"doc": doc, "title": "walk.ora", "uid": "pd1"})()

    packwright_mode.add_inker_document(ctx, inker_tab)
    assert [s.key for s in tab.doc.sprites()] == ["walk#frame0000", "walk#frame0001"]


def test_an_unknown_setting_is_a_framed_toast_rather_than_a_crash():
    """``dataclasses.replace`` raises ``TypeError`` for a field that does not
    exist, which the ``ValueError``-only clause let through as a crash."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, nonsense=1)
    assert ctx.toasts[-1][1] == "error"
    assert ctx.toasts[-1][0].startswith("That setting was not applied:")


def test_renaming_a_source_re_arms_the_pack():
    """The pane used to write onto the document, so the name changed in the
    list and the *layout* -- which is what the sidecar's ``filename`` comes
    from -- kept the old one until something else dirtied the pack."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "hero")
    assert tab.pack_dirty
    _pack(ctx, tab)
    assert [f.name for f in tab.layout.frames if f.key == "s0"] == ["hero"]


def test_a_rename_while_saving_is_ignored():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "hero")
    assert tab.doc.sources[0].name == "s0"


def test_a_name_the_sidecar_could_not_carry_is_refused_with_the_reason():
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "sub/hero")
    assert ctx.toasts[-1][1] == "error"
    assert ctx.toasts[-1][0].startswith("That name was not applied:")
    assert tab.doc.sources[0].name == "s0"


def test_removing_the_selected_source_clears_the_selection():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    uid = tab.doc.sources[0].uid
    state.selected = uid
    packwright_mode.remove_source(ctx, uid, tab)
    assert state.selected is None


def test_ask_open_focuses_an_already_open_atlas_instead_of_forking_a_second_tab(tmp_path):
    """The 2026-09-18 audit (second run, finding packwright-01):
    ``packwright-open`` adopted unconditionally while ``fileio.open_path``
    already guards -- the identical gap Clay closed in its own dialog arm on
    2026-09-12 (clay-02). Two tabs over one path race on save."""
    from realmspinner.studio.modes.packwright.engine.document import PackDoc

    ctx = FakeCtx()
    path = tmp_path / "atlas.rpack"
    path.write_bytes(b"")
    existing = _tab(ctx)
    existing.path = path

    packwright_mode.on_task_done(
        ctx, _Done("packwright-open", {"doc": PackDoc(), "path": str(path), "title": "Atlas"})
    )

    state = packwright_mode.ensure(ctx)
    assert len(state.docs) == 1
    assert state.active is existing


# --- saving -------------------------------------------------------------------


def test_a_save_writes_the_file_and_marks_the_document_clean(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_settings(padding=6)
    path = tmp_path / "atlas.rpack"
    packwright_mode.save_to(ctx, tab, path)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-save:{tab.uid}", ctx.result))

    assert path.exists() and not tab.dirty and not tab.saving
    assert tab.title == "atlas.rpack"
    assert rpack.read_rpack(path.read_bytes()).settings.padding == 6


def test_a_save_records_the_head_the_encode_wrote():
    ctx = FakeCtx()
    tab = _tab(ctx)
    head = tab.doc.history.head
    tab.doc.set_settings(padding=8)
    packwright_mode.on_task_done(
        ctx, _Done(f"packwright-save:{tab.uid}", {"head": head, "path": ""})
    )
    assert tab.dirty


def test_a_failed_save_clears_the_lock():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    packwright_mode.on_task_failed(ctx, _Done(f"packwright-save:{tab.uid}"))
    assert not tab.saving


def test_a_refused_submit_clears_the_lock_too(tmp_path):
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    packwright_mode.save_to(ctx, tab, tmp_path / "a.rpack")
    assert not tab.saving


# --- exporting ----------------------------------------------------------------


def test_a_grid_export_writes_the_png_the_json_and_a_tsx(tmp_path, monkeypatch):
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="grid")
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    packwright_mode.export_files(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-export:{tab.uid}", ctx.result))
    assert (tmp_path / "out.png").exists()
    assert (tmp_path / "out.json").exists()
    # A grid pack *is* a tileset, which is the whole payoff of the two modes
    # being genuinely different.
    assert (tmp_path / "out.tsx").exists()


def test_a_maxrects_export_writes_no_tsx(tmp_path, monkeypatch):
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="maxrects")
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    packwright_mode.export_files(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-export:{tab.uid}", ctx.result))
    assert (tmp_path / "out.json").exists()
    assert not (tmp_path / "out.tsx").exists()


def test_exporting_before_anything_is_packed_says_so():
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.export_files(ctx, tab)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert not tab.saving


# --- the guard ----------------------------------------------------------------


def test_the_guard_asks_once_for_however_many_are_dirty():
    """One question for all of them: asking per document would put the user in
    front of a queue they answered the same way each time."""
    ctx = FakeCtx()
    first = _tab(ctx)
    first.doc.set_settings(padding=6)
    calls: list[str] = []
    assert packwright_mode.guard(ctx, "quit", lambda: calls.append("go")) is False
    assert calls == []
    assert "One atlas has" in ctx.confirms.pending.message

    second = _tab(ctx)
    second.doc.set_settings(padding=8)
    packwright_mode.guard(ctx, "quit", lambda: calls.append("go"))
    assert "2 atlases have" in ctx.confirms.pending.message


def test_the_guard_is_silent_before_the_mode_has_ever_been_opened():
    ctx = FakeCtx()
    calls: list[str] = []
    packwright_mode.guard(ctx, "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.state.packwright is None


def test_closing_a_dirty_tab_asks_first():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_settings(padding=6)
    packwright_mode.close_tab(ctx, tab.uid)
    assert ctx.confirms.pending is not None
    ctx.confirms.pending.on_confirm()
    assert packwright_mode.ensure(ctx).get(tab.uid) is None


def test_closing_a_packwright_tab_forgets_its_remembered_column_count():
    """packwright-03, the 2026-09-08 audit: ``_last_columns`` is a module-level
    ``dict[tab.uid, int]`` remembering each tab's last explicit column count,
    and it is the one per-tab-uid cache in this segment with no matching
    release -- unlike ``packwright_textures.release_doc``, which
    ``close_tab``'s own release callback already calls for the texture cache
    keyed the same way. Every atlas tab ever given an explicit column count
    used to leave one entry behind for the life of the process."""
    from realmspinner.studio.modes.packwright.ui.panes import settings as packwright_settings

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_settings._last_columns[tab.uid] = 4

    packwright_mode.close_tab(ctx, tab.uid)

    assert tab.uid not in packwright_settings._last_columns


def test_closing_the_tab_a_pending_tileset_import_named_drops_it():
    """The 2026-09-16 audit: closing the tab a pending tile-set import
    belongs to (``tileset_import_uid``) used to leave the popup on screen
    describing a document that no longer exists -- with its texture and grid
    cache alive -- until a human noticed and cancelled it by hand. Fails
    against the unfixed code: ``close_tab`` never touches ``tileset_import``.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    state.tileset_import = ("C:/sheet.png", "sheet", np.zeros((4, 4, 4), dtype=np.uint8))
    state.tileset_import_uid = tab.uid
    state.tileset_import_open = True
    state.tileset_preview_key = ("stale", (32, 32), False, False)

    packwright_mode.close_tab(ctx, tab.uid)

    assert state.tileset_import is None
    assert state.tileset_import_uid == ""
    assert state.tileset_import_open is False
    assert state.tileset_preview_key is None


def test_closing_an_unrelated_tab_leaves_the_pending_tileset_import_alone():
    """The clear is scoped to the tab the import named -- closing some other
    open tab must not drop a popup the user is still answering."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    other = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    state.tileset_import = ("C:/sheet.png", "sheet", np.zeros((4, 4, 4), dtype=np.uint8))
    state.tileset_import_uid = tab.uid
    state.tileset_import_open = True

    packwright_mode.close_tab(ctx, other.uid)

    assert state.tileset_import is not None
    assert state.tileset_import_uid == tab.uid
    assert state.tileset_import_open is True


# --- keys ---------------------------------------------------------------------


def test_r_forces_a_repack(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0)
    assert packwright_mode.handle_key(ctx, event) is True
    assert tab.pack_dirty


def test_undo_re_arms_the_pack(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL)
    assert packwright_mode.handle_key(ctx, event) is True
    assert tab.pack_dirty


def test_ctrl_shift_z_redoes(monkeypatch):
    """What Inker, Clay and Plotter all accept. Ctrl+Y keeps working: this adds
    a spelling rather than replacing one."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, padding=6)
    tab.doc.undo()
    assert tab.doc.settings.padding != 6

    event = pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL | pygame.KMOD_SHIFT
    )
    assert packwright_mode.handle_key(ctx, event) is True
    assert tab.doc.settings.padding == 6


def test_delete_after_an_undo_does_not_crash(monkeypatch):
    """Ctrl+Z after adding a source left ``selected`` naming a detached uid,
    and Delete then raised a bare ``KeyError`` into the pygame key dispatch."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    state = packwright_mode.ensure(ctx)
    added = tab.doc.add_source(_sprite("new"))
    state.selected = added.uid

    packwright_mode.handle_key(
        ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL)
    )
    assert state.selected is None

    packwright_mode.handle_key(
        ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0)
    )
    assert len(tab.doc.sources) == 1


def test_an_undo_that_keeps_the_source_keeps_the_selection(monkeypatch):
    """Narrower than Plotter's unconditional clear: a source's uid survives
    every step but the one that detaches it, so an undone *rename* has no
    business deselecting the row it renamed."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    uid = tab.doc.sources[0].uid
    tab.doc.rename_source(uid, "hero")
    state.selected = uid

    packwright_mode.handle_key(
        ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL)
    )
    assert state.selected == uid


def test_delete_removes_the_selected_source(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    state.selected = tab.doc.sources[1].uid
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE, mod=0)
    assert packwright_mode.handle_key(ctx, event) is True
    assert len(tab.doc.sources) == 2


def test_a_key_release_is_never_consumed():
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_r, mod=0)
    assert packwright_mode.handle_key(ctx, up) is False


def test_recent_files_persist():
    ctx = FakeCtx()
    doc = packwright_mode.new_document(ctx).doc
    path = Path("/tmp/atlas.rpack")
    packwright_mode.adopt(ctx, doc, path=path)
    assert packwright_mode.recent_paths(ctx) == [str(path)]


# --- crash recovery -------------------------------------------------------------


def test_a_recovered_atlas_reads_dirty_and_close_asks(tmp_path):
    """``read_rpack`` hands back a document already marked saved, and
    ``PackTab.dirty`` delegates to the document -- so an adopt that only wrote
    ``tab.saved_head`` produced a *clean* recovered tab: one unprompted close
    skipped the confirm and ``drop()`` deleted the journal copy, the only
    surviving copy of the work.

    ``_journal_adopt`` only submits now (packwright-01, below) -- the tab
    appears once ``on_task_done`` lands the result, ``FakeCtx.submit``'s
    inline-run shape.
    """
    seed = FakeCtx()
    source = packwright_mode.new_document(seed)
    source.doc.add_source(_sprite("s0"))
    path = tmp_path / "atlas.rpack"
    path.write_bytes(rpack.rpack_bytes(source.doc))

    ctx = FakeCtx()
    assert packwright_mode._journal_adopt(ctx, path, {"title": "atlas"}) is True
    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    tab = ctx.state.packwright.docs[-1]
    assert tab.dirty is True, "recovered work is unsaved by definition"

    packwright_mode.close_tab(ctx, tab.uid)
    assert ctx.confirms.pending is not None, "closing recovered work must ask"
    assert ctx.state.packwright.get(tab.uid) is not None, "still open until answered"


class _ThreadedCtx(FakeCtx):
    """``submit`` on a real worker thread, joined -- so the test sees the task
    half run where it would run, and a regression that moves the decode back
    onto the calling thread shows up as that thread's name. The technique
    ``tests/test_frame_thread_doors.py`` uses for every other frame-thread
    door (``_Threaded`` there); kept local here since this file owns
    Packwright's fix and that file's own door list is not this session's to
    edit.
    """

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        box: dict[str, Any] = {}

        def go() -> None:
            box["result"] = run(*args)

        worker = threading.Thread(target=go, name="packwright-mode-test-worker")
        worker.start()
        worker.join()
        self.result = box["result"]
        return True


def test_packwright_journal_adopt_reads_and_decodes_on_a_task_not_the_frame_thread(
    tmp_path, monkeypatch
):
    """packwright-01 (the 2026-09-11 audit): ``_journal_adopt`` used to read
    the ``.rpack`` and decompress every sprite PNG synchronously, on whatever
    thread called it -- the frame thread, since the Recover button in
    ``panes/landing.py`` calls ``journal.take`` -> ``journal.adopt`` ->
    ``provider.adopt`` with no ``ctx.submit`` anywhere in that chain (unlike
    Clay's and Inker's own providers, which already defer the same shape of
    work). Measured at 137.7 ms for a 300-sprite/128px atlas -- an ~8-frame
    freeze on one click.

    Proved behaviourally: the decode runs on a real worker thread (not the
    thread that called ``_journal_adopt``), and nothing is adopted into
    ``ctx.state.packwright`` until the task's result is handed to
    ``on_task_done`` -- a name check alone would still pass if someone kept
    ``ctx.submit`` in the call but left the actual read+decode running inline
    ahead of it.
    """
    seed = FakeCtx()
    source = packwright_mode.new_document(seed)
    source.doc.add_source(_sprite("s0"))
    path = tmp_path / "atlas.rpack"
    path.write_bytes(rpack.rpack_bytes(source.doc))

    threads: list[str] = []
    real_read = rpack.read_rpack

    def spy(data: bytes) -> Any:
        threads.append(threading.current_thread().name)
        return real_read(data)

    monkeypatch.setattr(rpack, "read_rpack", spy)

    ctx = _ThreadedCtx()
    assert packwright_mode._journal_adopt(ctx, path, {"title": "atlas"}) is True
    # Submitted, not adopted: the read has run (on the worker thread, joined
    # above), but no tab exists until on_task_done lands the result.
    assert threads == ["packwright-mode-test-worker"]
    assert ctx.state.packwright.docs == []

    packwright_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    tab = ctx.state.packwright.docs[-1]
    assert tab.title == "atlas (recovered)"
    assert tab.dirty is True
    assert tab.journal_name == path.name


def test_a_recovered_atlas_that_will_not_parse_says_so(tmp_path):
    """Through ``journal.adopt_failed`` (2026-09-05's sentence every provider
    says), where a raise here would arrive as an *error* toast no other
    mode's copy raises. Clay's sibling:
    ``test_frame_thread_doors.py::test_a_recovered_clay_model_that_will_not_parse_says_so``.
    """
    path = tmp_path / "bad.rpack"
    path.write_bytes(b"not a zip")
    assert packwright_mode._load_recovery(path, {}) is None


def test_export_library_refuses_while_the_pack_is_behind():
    """Unlike the file export, whose PNG and sidecar both derive from the
    landed pack, this pairs the landed atlas with the *current* document -- so
    while an edit is unpacked the two would describe different atlases."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    tab.pack_dirty = True

    packwright_mode.export_library(ctx, tab)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert not any(key.startswith("packwright-library") for key in ctx.submitted)


def test_adding_a_library_asset_with_no_atlas_starts_one(tmp_path, monkeypatch):
    """The Library offers this for any asset carrying an ``input.png`` and
    cannot know whether an atlas is open, so the error toast was an offer taken
    back. Unlike a map, an atlas has no numbers that cannot be taken back
    later, so one is simply made -- and the user is taken to it, because a new
    empty atlas they cannot see is the same dead end by another route."""
    from PIL import Image

    png = tmp_path / "input.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(png)
    monkeypatch.setattr("realmspinner.service.files.job_dir_file", lambda svc, job_id, name: png)

    ctx = FakeCtx()
    assert packwright_mode.active(ctx) is None

    packwright_mode.add_job_source(ctx, {"id": "j1", "name": "chest"})

    tab = packwright_mode.active(ctx)
    assert tab is not None
    assert not ctx.toasts
    assert ctx.state.mode == "packwright"
    assert ctx.submitted == [f"packwright-add:{tab.uid}:j1"]


class _DedupingCtx(FakeCtx):
    """A ctx whose ``submit`` refuses a key already in flight, as the real
    runner does, and hands its results back through ``on_task_done``.

    The plain ``FakeCtx`` above runs every submit inline and keeps no keys, so
    it cannot see a refusal at all -- which is why a multi-file drop looked
    fine here for as long as it was broken in the app.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[str, Any]] = []
        self.refused: list[str] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if any(key == in_flight for in_flight, _ in self.pending):
            self.refused.append(key)
            return False
        self.pending.append((key, run(*args)))
        return True

    def deliver(self) -> None:
        for key, result in self.pending:
            packwright_mode.on_task_done(
                self, type("Done", (), {"key": key, "result": result})()
            )
        self.pending.clear()


def test_dropping_several_files_adds_every_one(tmp_path):
    """pygame raises one ``DROPFILE`` per file, so a drag of three PNGs is
    three calls in one pump. Under a per-tab key the runner refused the second
    and third and nothing read the refusal, so a multi-file drop added the
    first file and said "Added 1 sprite(s)"."""
    from PIL import Image

    paths = []
    for index, colour in enumerate([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]):
        path = tmp_path / f"tile{index}.png"
        Image.new("RGBA", (4, 4), colour).save(path)
        paths.append(path)

    ctx = _DedupingCtx()
    tab = packwright_mode.new_document(ctx)
    for path in paths:
        packwright_mode.add_source_paths(ctx, [path])

    assert ctx.refused == [], "every distinct drop got its own key"
    ctx.deliver()
    # The key is ``sources.file_key`` since 2026-09-04: the stem plus a digest
    # of where the file came from. It was ``str(path)``, which wrote the
    # author's directory layout into the shared ``.rpack``.
    from realmspinner.studio.modes.packwright.engine.sources import file_key

    assert [s.key for s in tab.doc.sprites()] == [file_key(p) for p in paths]
    assert not any(str(tmp_path) in s.key for s in tab.doc.sprites())


def test_a_rendered_sheet_handoff_is_not_silently_dropped_while_a_manual_tileset_pick_is_open(
    tmp_path, monkeypatch
):
    """packwright-02: ``ask_add_tileset`` and ``add_rendered_sheet`` used to
    submit under the identical bare key ``packwright-tileset:{uid}``, and
    neither call site inspects ``ctx.submit``'s return -- so a Troupe handoff
    landing while the manual picker's OS dialog task was still "in flight"
    under that key was refused, silently: manual submit True, handoff False,
    no toast. Reproduced with ``_DedupingCtx``, the only fake here that can
    see a refusal at all."""
    from PIL import Image

    from realmspinner.studio import dialogs

    png = tmp_path / "sheet.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(png)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: png)
    monkeypatch.setattr(
        "realmspinner.service.sheets.get_sheet", lambda svc, job_id, sheet_id: {"name": "walk"}
    )
    monkeypatch.setattr("realmspinner.service.sheets.sheet_png", lambda svc, job_id, sheet_id: png)
    monkeypatch.setattr(
        "realmspinner.studio.modes.inker.mode.sheet_grid", lambda record: ((4, 4), 1)
    )

    ctx = _DedupingCtx()
    packwright_mode.new_document(ctx)

    packwright_mode.ask_add_tileset(ctx)  # the manual picker's task, still "in flight"
    packwright_mode.add_rendered_sheet(ctx, "j1", "sheet1")  # the handoff

    assert ctx.refused == [], "the handoff must not collide with the manual picker's key"


def test_a_second_tileset_landing_does_not_silently_replace_one_already_parked_under_an_open_popup():  # noqa: E501
    """packwright-01, the 2026-09-08 audit: ``on_task_done``'s
    ``"packwright-tileset"`` branch used to adopt whatever landed
    unconditionally, even while an earlier import's popup was still open and
    unconfirmed -- silently swapping the sheet under it and dropping
    ``tileset_import_open`` back to False, so the pane's own "open once a new
    import lands" check (``modes/packwright/ui/panes/sources.py``) reopened the popup
    over completely different pixels next frame, with no toast and no visible
    sign anything had changed."""
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)

    first = np.zeros((4, 4, 4), dtype=np.uint8)
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("first.png", "first", first), "uid": tab.uid},
        ),
    )
    assert packwright_mode.ensure(ctx).tileset_import[1] == "first"

    # The pane's own draw-loop behaviour: the popup opens once, the first
    # time it sees a pending import, and stays open until confirmed or
    # cancelled.
    state = packwright_mode.ensure(ctx)
    state.tileset_import_open = True

    second = np.ones((4, 4, 4), dtype=np.uint8)
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("second.png", "second", second), "uid": tab.uid},
        ),
    )

    assert state.tileset_import[1] == "first", "the parked import must not be swapped"
    assert state.tileset_import_open is True, "the open popup must not be reset"
    assert ctx.toasts, "a dropped second import must say so"


def test_the_same_drop_twice_over_still_dedupes(tmp_path):
    """The half of the key worth keeping: one file decoded twice at once is
    one decode, and the second press was not a second intention."""
    from PIL import Image

    path = tmp_path / "tile.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(path)

    ctx = _DedupingCtx()
    packwright_mode.new_document(ctx)
    packwright_mode.add_source_paths(ctx, [path])
    packwright_mode.add_source_paths(ctx, [path])

    assert len(ctx.refused) == 1


def test_an_inker_document_with_no_atlas_starts_one_too():
    """The same door from Inker's bridge, which is where this is now offered
    from -- Packwright's sources pane could already pull a document in, and a
    push from the near side must not refuse for want of an atlas."""
    from realmspinner.kernels.pixel.document import Document

    ctx = FakeCtx()
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = (255, 0, 0, 255)
    inker_tab = type("T", (), {"doc": doc, "title": "walk.ora", "uid": "pd1"})()

    packwright_mode.add_inker_document(ctx, inker_tab)

    tab = packwright_mode.active(ctx)
    assert tab is not None
    assert [s.key for s in tab.doc.sprites()] == ["walk#layer00:Background"]


# --- W0.2: the empty states told the truth for the first time ---------------


def test_no_pane_promises_a_pack_button_that_does_not_exist() -> None:
    """Two empty states read "Add images and press Pack." Packing is automatic
    -- run from the centre pane's pump -- and the only manual trigger was a
    bare ``R`` nothing on screen mentioned. An instruction naming a control
    that has never existed is worse than no instruction at all."""
    from _panes import pane_files

    for name in ("packwright_bridge.py", "packwright_items.py"):
        source = pane_files()[name].read_text(encoding="utf-8")
        assert "press Pack" not in source, name


def test_the_settings_pane_offers_the_repack_r_already_did() -> None:
    """``R`` set ``pack_dirty`` inline and nothing on screen said so. The
    button and the key are one verb now, not two that happen to agree."""
    from _panes import pane_files

    from realmspinner.studio.modes.packwright import mode as mode

    assert hasattr(mode, "request_repack")
    source = pane_files()["packwright_settings.py"].read_text(encoding="utf-8")
    assert "Repack now" in source
    assert "packwright_mode.request_repack(" in source


# --- W1.8: Automatic, not a magic zero ---------------------------------------


def test_columns_offer_automatic_instead_of_a_magic_zero(monkeypatch):
    """The Columns field used to store ``int(columns) or None``, so 0 meant
    "auto" and nothing on screen said so. Now there is an Automatic checkbox:
    on, the field is disabled and the document holds ``None``; off, the field
    is live and starts at the user's own last explicit count -- or, nobody has
    set one yet, at the count Automatic just landed on. The document field's
    shape is unchanged (``PackSettings.columns: int | None``); only the
    control moved. Pressed for real, through a real imgui frame, for the
    reason ``test_context_controls`` gives: a control wired to nothing passes
    every test that calls the setter directly."""
    from _ui_context import imgui_context

    from realmspinner.studio import probe, widgets
    from realmspinner.studio.modes.packwright.ui.panes import settings as packwright_settings

    with imgui_context(monkeypatch) as ui:
        ctx = FakeCtx()
        tab = _tab(ctx)  # three sources, grid mode, columns still None
        _pack(ctx, tab)
        automatic_result = tab.layout.columns
        assert automatic_result >= 1

        real_toggle = widgets.toggle
        captured: dict[str, tuple[float, float, float, float]] = {}

        def spy_toggle(label, value, *, tag=None, tooltip=""):
            # ``widgets.toggle`` draws through a raw ``invisible_button``, so
            # it never reaches ``probe`` -- see ``probe``'s own module
            # docstring. Its rect is still the last item the moment this
            # returns, which is what the census gets for every other control.
            changed, out = real_toggle(label, value, tag=tag, tooltip=tooltip)
            if label == "Automatic":
                low = ui.get_item_rect_min()
                high = ui.get_item_rect_max()
                captured["Automatic"] = (low.x, low.y, high.x - low.x, high.y - low.y)
            return changed, out

        monkeypatch.setattr(widgets, "toggle", spy_toggle)

        def build():
            packwright_settings.draw(ctx)

        def frame(pos=(-100.0, -100.0), down=False):
            io = ui.get_io()
            io.add_mouse_pos_event(pos[0], pos[1])
            io.add_mouse_button_event(0, down)
            probe.begin_frame()
            ui.new_frame()
            ui.set_next_window_size((900.0, 900.0))
            ui.set_next_window_pos((0.0, 0.0))
            ui.begin("##host")
            build()
            ui.end()
            ui.end_frame()
            return list(probe.FRAME_CONTROLS)

        def centre_of(rect_):
            x, y, w, h = rect_
            return (x + w * 0.5, y + h * 0.5)

        def click(pos):
            frame(pos, down=True)
            frame(pos, down=False)

        def columns_rect():
            controls = frame()
            found = [c for c in controls if c.label == "##Columns"]
            assert found, [c.label for c in controls]
            return found[0].rect

        # Starts automatic: the document already defaults there.
        assert tab.doc.settings.columns is None

        # A real drag attempt on the disabled field must not move it.
        cx, cy, cw, ch = columns_rect()
        frame((cx + 4, cy + ch * 0.5), down=True)
        frame((cx + cw * 0.75, cy + ch * 0.5), down=True)
        frame((cx + cw * 0.75, cy + ch * 0.5), down=False)
        assert tab.doc.settings.columns is None, "a disabled field must not accept a drag"

        # Turn Automatic off: with nothing remembered, the field starts at the
        # count Automatic itself just landed on.
        click(centre_of(captured["Automatic"]))
        assert tab.doc.settings.columns == automatic_result

        # Drag the now-live field to some other explicit count.
        cx, cy, cw, ch = columns_rect()
        frame((cx + 4, cy + ch * 0.5), down=True)
        frame((cx + cw, cy + ch * 0.5), down=True)
        frame((cx + cw, cy + ch * 0.5), down=False)
        explicit = tab.doc.settings.columns
        assert explicit is not None and explicit != automatic_result, (
            "the drag on the now-enabled field did not change anything"
        )
        assert packwright_settings._last_columns[tab.uid] == explicit

        # Automatic back on: the document goes back to None...
        click(centre_of(captured["Automatic"]))
        assert tab.doc.settings.columns is None

        # ...and off again hands back the user's own count, not the automatic
        # result the field started at the first time.
        click(centre_of(captured["Automatic"]))
        assert tab.doc.settings.columns == explicit


# --- bridge pane ----------------------------------------------------------


def test_packed_why_says_packing_while_the_first_pack_runs():
    """The 2026-09-20 audit, finding packwright-01: ``bridge.py``'s
    ``packed_why`` fell back straight to "Nothing is packed yet" whenever
    ``tab.layout is None``, with no ``tab.packing`` check -- while its two
    siblings in this mode (``items.py``'s "Packing...", ``preview.py``'s
    ``elif tab.packing`` branch) both make that distinction. A user who just
    dropped sprites and is still waiting on the first pack read the export
    button's reason as the drop having produced nothing at all."""
    from realmspinner.studio.modes.packwright.ui.panes import bridge as packwright_bridge

    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    assert tab.layout is None and tab.atlas is None and tab.pack_stale_why == ""

    tab.packing = True
    assert packwright_bridge._packed_why(tab) == "Packing..."

    tab.packing = False
    assert packwright_bridge._packed_why(tab) == (
        "Nothing is packed yet. Add images -- packing runs by itself."
    )
