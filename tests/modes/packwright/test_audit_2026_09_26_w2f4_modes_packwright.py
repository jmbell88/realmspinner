"""Regression tests closing the 2026-09-26 audit's Packwright findings (w2f4).

One file per the fixer brief's naming rule, covering every finding this pass
closed across ``mode.py``, ``fileio.py``, ``state.py``, the bridge/items panes
and the ``rpack``/``texturepacker``/``trim`` engine modules. Each test's name is
the claim, and each failed against the unfixed code before the corresponding
fix landed (see the paired PR/report for the pasted failures).
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import layout as packwright_layout
from realmspinner.studio.modes.packwright.engine import rpack, texturepacker
from realmspinner.studio.modes.packwright.engine.sources import Sprite
from realmspinner.studio.modes.packwright.engine.trim import trim_rect

from .test_packwright_mode import FakeCtx, _Done, _pack, _tab
from .test_rpack import _doc, _rewrite


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- packwright-mode-01: a stale pack landing after the last source went ------


def test_a_pack_landing_after_the_last_source_was_removed_is_dropped():
    """packwright-mode-01. A pack submitted while a source existed that lands
    *after* the last source is removed used to install the stale atlas anyway:
    ``request_pack``'s own empty-sources branch already clears
    ``layout``/``atlas``/``pack_dirty`` the moment the document goes empty, but
    ``adopt_pack`` deliberately never touches ``pack_dirty`` (an edit mid-pack
    must survive the landing) -- so the stale result re-installed the old
    atlas over an empty document with ``pack_dirty`` left at False and nothing
    left to re-arm a repack."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)

    packwright_mode.request_pack(ctx, tab)
    stale_result = ctx.result
    assert isinstance(stale_result, dict) and stale_result.get("layout") is not None

    # The last source is removed while that pack is (conceptually) still
    # running.
    packwright_mode.remove_source(ctx, tab.doc.sources[0].uid, tab)
    assert tab.pack_dirty is True

    # The centre pane's pump runs before the task lands: request_pack's own
    # empty-sources branch self-heals first.
    packwright_mode.request_pack(ctx, tab)
    assert tab.layout is None and tab.atlas is None and tab.pack_dirty is False

    # The stale task then lands.
    packwright_mode.on_task_done(ctx, _Done(f"packwright-pack:{tab.uid}", stale_result))

    assert tab.layout is None, "the stale atlas must not be installed over an empty document"
    assert tab.atlas is None
    assert tab.pack_dirty is False
    assert not tab.packing


# --- packwright-mode-02: a failed add/tileset/pack must not unlock a save ----


def test_a_failed_pack_does_not_unlock_a_tab_that_is_still_saving():
    """packwright-mode-02. ``on_task_failed`` used to clear ``tab.saving`` for
    *any* failed task keyed to the tab, including a pack -- which never sets
    ``saving`` in the first place -- so a pack that failed while a real save
    (a different task, under a different key) was still writing unlocked the
    tab mid-encode."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    packwright_mode.on_task_failed(
        ctx, _Done(f"packwright-pack:{tab.uid}", message="does not fit")
    )

    assert tab.saving is True, "a concurrent save must not be unlocked by an unrelated failure"
    assert not tab.packing
    assert tab.pack_error == "does not fit"


def test_a_failed_add_does_not_unlock_a_tab_that_is_still_saving():
    """The other half of packwright-mode-02: a failed picker add or tile-set
    decode is the other shape that reaches ``on_task_failed`` with no
    ``saving`` of its own to clear."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    packwright_mode.on_task_failed(
        ctx, _Done(f"packwright-add:{tab.uid}", message="could not decode")
    )

    assert tab.saving is True


# --- packwright-mode-03: "Atlas + JSON" must refuse a stale/in-flight pack ---


def test_export_files_refuses_while_an_edit_is_unpacked(tmp_path, monkeypatch):
    """packwright-mode-03. ``export_library`` already refuses while
    ``pack_dirty``/``packing`` is set -- pairing the landed atlas with a
    document that has since moved on -- but ``export_files`` ("Atlas + JSON")
    had no such check, so Ctrl+Shift+E wrote a PNG+JSON pair for the atlas
    from before an edit while the open document already disagreed with it."""
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    # An edit re-arms the pack; the atlas on screen is still the one from
    # before it.
    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "hero")
    assert tab.pack_dirty is True

    packwright_mode.export_files(ctx, tab)

    assert not (tmp_path / "out.png").exists(), "must not export the stale atlas"
    errors = [text for text, level in ctx.toasts if level == "error"]
    assert any("Still packing" in text for text in errors), ctx.toasts


def test_the_export_files_button_greys_while_an_edit_is_unpacked():
    """packwright-mode-03's other half: the bridge pane's own gate must match
    the refusal above, so the button reads grey rather than firing a refusal
    toast on press."""
    from realmspinner.studio.modes.packwright.ui.panes import bridge as packwright_bridge

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    assert packwright_bridge._packed(tab) is True

    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "hero")
    assert tab.pack_dirty is True
    assert packwright_bridge._packed(tab) is False

    tab.pack_dirty = False
    tab.packing = True
    assert packwright_bridge._packed(tab) is False


# --- packwright-mode-05: the pack-failure remedy must fit the mode ----------


def test_the_pack_failure_remedy_does_not_suggest_trimming_for_a_grid_pack():
    """packwright-mode-05. ``layout.grid_layout`` always packs with trim off
    (see its own module docstring: "a grid pack never trims, whatever trim
    says"), so telling the user to turn trimming on after a grid pack failure
    is a remedy that does nothing."""
    from realmspinner.studio.modes.packwright.ui.panes import items as packwright_items

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="grid")

    remedy = packwright_items._pack_error_remedy(tab)
    assert "trim" not in remedy.lower(), remedy
    assert "max size" in remedy

    packwright_mode.set_settings(ctx, tab, mode="maxrects")
    remedy = packwright_items._pack_error_remedy(tab)
    assert "trim" in remedy.lower(), "maxrects can still trim, and the remedy should say so"


# --- packwright-mode-06: set_pivot must refuse, not silently drop -----------


def test_set_pivot_refuses_while_the_tab_is_saving_rather_than_dropping_silently():
    """packwright-mode-06. Every sibling in this file (``remove_source``,
    ``rename_source``, ``set_settings``, the add doors) gates on ``tab.busy``
    and toasts the shared refusal sentence; ``set_pivot`` gated on
    ``tab.saving`` directly and returned with no word at all."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    before_head = tab.doc.history.head

    packwright_mode.set_pivot(ctx, tab, tab.doc.sources[0].uid, (0.5, 0.5))

    assert tab.doc.history.head == before_head, "the pivot must not have been applied"
    errors = [text for text, level in ctx.toasts if level == "error"]
    assert errors, "a refusal while saving must say so, not drop the edit silently"


# --- packwright-packer-01: an infinite number must not crash the open -------


def test_read_rpack_refuses_an_infinite_number_in_settings_or_slice_bounds_by_name():
    """packwright-packer-01. A manifest's ``1e999`` parses through
    ``json.loads`` as Python's own ``inf`` -- no exception there -- and it is
    the ``int(inf)`` that follows, in ``_settings_from`` or ``_meta_from``,
    that raises ``OverflowError`` -- past the ``except (TypeError, ValueError)``
    clauses that were the only guard. Both must come back as an ordinary named
    refusal instead of an uncaught ``OverflowError``. ``float("inf")`` here is
    what ``1e999`` becomes the moment ``json.loads`` reads it -- the same
    downstream value, produced without hand-writing the archive's bytes."""

    def bad_padding(manifest):
        manifest["settings"]["padding"] = float("inf")

    with pytest.raises(ValueError, match="malformed") as caught:
        rpack.read_rpack(_rewrite(_doc(), bad_padding))
    assert not isinstance(caught.value, OverflowError)

    def bad_slice_bounds(manifest):
        manifest["sources"][0]["slices"] = [
            {"name": "a", "bounds": {"x": float("inf"), "y": 0, "w": 1, "h": 1}}
        ]

    with pytest.raises(ValueError, match="malformed") as caught:
        rpack.read_rpack(_rewrite(_doc(), bad_slice_bounds))
    assert not isinstance(caught.value, OverflowError)


# --- packwright-packer-02: a non-finite pivot must not become invalid JSON --


def test_read_rpack_refuses_a_non_finite_pivot_by_name():
    """packwright-packer-02. ``float()`` accepts ``inf``/``nan`` with no
    exception, and JSON itself has literals for both that ``json.loads``
    reads -- so a pivot of either used to open clean and ride into the
    exported sidecar as the literal token ``Infinity``/``NaN``, which is not
    valid JSON."""

    for bad in (float("inf"), float("-inf"), float("nan")):

        def add_pivot(manifest, bad=bad):
            manifest["sources"][0]["pivot"] = {"x": bad, "y": 0.5}

        with pytest.raises(ValueError, match="malformed") as caught:
            rpack.read_rpack(_rewrite(_doc(), add_pivot))
        assert not isinstance(caught.value, (OverflowError, TypeError))


def test_tp_bytes_refuses_a_non_finite_pivot_rather_than_write_invalid_json():
    """packwright-packer-02's other door: even if a non-finite pivot ever
    reached ``tp_bytes`` by some other path than ``rpack.read_rpack``, the
    sidecar writer itself must refuse rather than silently emit
    ``allow_nan=True``'s ``Infinity``/``NaN`` tokens."""
    frame = packwright_layout.Frame(
        key="s0",
        name="s0",
        x=0,
        y=0,
        w=4,
        h=4,
        trim=(0, 0, 4, 4),
        source_w=4,
        source_h=4,
        pivot=(float("inf"), 0.5),
    )
    result = packwright_layout.Layout(
        width=4, height=4, mode="maxrects", padding=0, extrude=0, frames=(frame,)
    )
    with pytest.raises(ValueError):
        texturepacker.tp_bytes(result, image_name="out.png")


# --- packwright-packer-03: a grid pack of blank sprites must not shrink -----


def test_trim_rect_keeps_the_full_canvas_for_a_blank_sprite_when_untrimmed():
    """packwright-packer-03. ``trim_rect`` used to check ``box is None``
    before ``enabled``, so a fully transparent sprite collapsed to the 1x1
    empty rectangle even with trimming off."""
    blank = np.zeros((16, 24, 4), dtype=np.uint8)
    assert trim_rect(blank, enabled=False) == (0, 0, 24, 16, True)
    # Trimming on is unaffected: still the 1x1 empty box (``EMPTY_SIZE``).
    assert trim_rect(blank, enabled=True) == (0, 0, 1, 1, True)


def test_a_grid_pack_of_all_blank_sprites_keeps_their_own_cell_size():
    """The end-to-end shape the finding actually reported: an all-blank grid
    pack's ``.tsx`` described 1x1 tiles for sprites that were really 16x16,
    because grid mode always packs with trim off (``layout.grid_layout``)."""
    blank = np.zeros((16, 16, 4), dtype=np.uint8)
    sprites = [Sprite(key=f"s{i}", name=f"s{i}", pixels=blank) for i in range(2)]

    result = packwright_layout.grid_layout(sprites, packwright_layout.PackSettings(mode="grid"))

    assert result.cell_w == 16
    assert result.cell_h == 16
