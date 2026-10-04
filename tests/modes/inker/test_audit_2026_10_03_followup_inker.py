"""The 2026-10-03 audit's Inker follow-ups: inker-41's Export doors, the tile
conversion and the layer-properties popup's greyed controls, an alpha-locked
landing, and the sheet-merge refusal's sentence.

Each test's name is the claim, and each fails against the code as it stood
before the fix.
"""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.animation import Tag
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.inker import export as inker_export
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import sheet as inker_sheet
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import generate as inker_generate
from realmspinner.studio.modes.inker.ui.panes import menu as inker_menu_pane
from realmspinner.studio.modes.inker.ui.panes import tiles as inker_tiles

RED = (255, 0, 0, 255)


def _ctx_for(state):
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    stored: dict = {}
    settings = SimpleNamespace(
        get=lambda key: stored.get(key, {}), set=lambda key, value: stored.update({key: value})
    )
    return SimpleNamespace(state=app, toast=app.toast, settings=settings)


def _session(*, frames=3, layers=2):
    doc = inker.Document.blank(8, 8)
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
    for layer in doc.stack:
        layer.pixels[2:6, 2:6] = RED
    doc.invalidate_all()
    if frames > 1:
        doc.ensure_animation()
        for _ in range(frames - 1):
            doc.add_frame()
        doc.anim.tags.append(Tag(name="walk", start=0, end=1))
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    return _ctx_for(state), state, tab


def _toasts(ctx):
    return [toast.text for toast in getattr(ctx.state, "toasts", [])]


def _open_transform(ctx, state, tab):
    tab.doc.select_all()
    inker_mode.begin_transform(ctx, tab)
    assert state.transforming and tab.doc.floating is not None, "fixture: a transform is open"


# --- inker-41: the five Export doors ------------------------------------------


def test_door_state_greys_every_door_under_an_open_transform_and_says_why():
    ctx, state, tab = _session()
    state.transforming = True
    for door in inker_export.doors():
        enabled, reason = inker_export.door_state(door, tab, state)
        assert not enabled, door.key
        assert "free transform" in reason, (door.key, reason)


def test_door_state_is_unchanged_without_a_state_or_with_a_quiet_one():
    ctx, state, tab = _session()
    sheet = next(d for d in inker_export.doors() if d.key == "sheet")
    assert inker_export.door_state(sheet, tab) == (True, "")
    assert inker_export.door_state(sheet, tab, state) == (True, "")
    state.transforming = True
    # The document's own refusals still come first: nothing is open, so the
    # transform is not the reason.
    assert inker_export.door_state(sheet, None, state) == (False, inker_export.NO_DOCUMENT_WHY)


def test_the_bridge_and_the_menu_pass_the_state_to_door_state():
    from realmspinner.studio import menus

    assert "door_state(door, tab, state)" in inspect.getsource(inker_generate._exports)
    call = inspect.getsource(menus._inker_export_specs).split("door_state(")[1].split(")")[0]
    assert "state" in call


def test_the_file_menu_greys_the_five_exports_under_an_open_transform():
    from realmspinner.studio import menus

    ctx, state, tab = _session()
    ctx.state.mode = "inker"
    state.transforming = True

    rows = menus._inker_export_specs(ctx, [])

    assert len(rows) == 5
    for row in rows:
        assert not row.enabled and "free transform" in row.disabled_reason, row.identity


def _doors_refusing(monkeypatch, ctx, state, tab):
    started = []
    monkeypatch.setattr(inker_mode, "_start", lambda *a, **k: started.append(a))
    return started


@pytest.mark.parametrize(
    "run",
    [
        lambda ctx, tab: inker_export.export_png(ctx, tab),
        lambda ctx, tab: inker_export.export_sheet(ctx, tab),
        lambda ctx, tab: inker_export.export_gif(ctx, tab),
        lambda ctx, tab: inker_export.export_pngs(ctx, tab),
        lambda ctx, tab: inker_export.export_per_tag(ctx, tab, "sheet"),
        lambda ctx, tab: inker_export.export_per_layer(ctx, tab, "sheet"),
        lambda ctx, tab: inker_export.export_range(ctx, tab, "sheet", (0, 1)),
        lambda ctx, tab: inker_export.export_tag(ctx, tab, "gif", 0),
        lambda ctx, tab: inker_export.open_door(ctx, tab, "sheet"),
        lambda ctx, tab: inker_export.open_door(ctx, tab, "per-layer"),
        lambda ctx, tab: inker_export.repeat_export(ctx, tab),
    ],
    ids=[
        "png", "sheet", "gif", "pngs", "per_tag", "per_layer", "range", "tag",
        "open_door_sheet", "open_door_per_layer", "repeat",
    ],
)
def test_no_export_door_commits_the_float_while_a_transform_is_open(monkeypatch, run):
    ctx, state, tab = _session()
    started = _doors_refusing(monkeypatch, ctx, state, tab)
    # A recorded export so Repeat has something to repeat.
    tab.export_kind, tab.export_dest = "png", inker_export.Path("x.png")
    _open_transform(ctx, state, tab)

    run(ctx, tab)

    assert tab.doc.floating is not None, "the export committed the free transform's float"
    assert state.transforming is True
    assert state.export is None and not tab.saving and not started
    assert inker_export.TRANSFORM_WHY in _toasts(ctx)


def test_export_slices_refuses_under_a_transform_before_it_looks_at_the_slices(monkeypatch):
    ctx, state, tab = _session()
    started = _doors_refusing(monkeypatch, ctx, state, tab)
    _open_transform(ctx, state, tab)

    inker_export.export_slices(ctx, tab)

    assert _toasts(ctx) == [inker_export.TRANSFORM_WHY], _toasts(ctx)
    assert tab.doc.floating is not None and not started


def test_the_doors_still_run_once_the_transform_is_closed(monkeypatch):
    ctx, state, tab = _session()
    started = _doors_refusing(monkeypatch, ctx, state, tab)

    inker_export.export_png(ctx, tab)
    inker_export.export_sheet(ctx, tab)

    assert state.export is not None and len(started) == 1
    assert inker_export.TRANSFORM_WHY not in _toasts(ctx)


# --- tiles and the layer-properties popup --------------------------------------


def test_convert_to_tilemap_is_greyed_under_an_open_transform_with_a_sentence():
    ctx, state, tab = _session(frames=1, layers=1)
    assert inker_tiles.can_convert(state, tab) is True
    assert inker_tiles.convert_gate(state, tab) == (True, "")

    state.transforming = True
    assert inker_tiles.can_convert(state, tab) is False
    enabled, reason = inker_tiles.convert_gate(state, tab)
    assert not enabled and "free transform" in reason

    state.transforming = False
    tab.saving = True
    enabled, reason = inker_tiles.convert_gate(state, tab)
    assert not enabled and reason == inker_tiles.BUSY_WHY


def test_the_convert_button_and_the_layer_properties_popup_say_why_they_are_greyed():
    popup = inspect.getsource(inker_tiles._tile_size_popup)
    assert "begin_disabled(tab.busy)" not in popup
    assert "convert_gate(" in popup
    properties = inspect.getsource(inker_menu_pane._properties_popup)
    assert "begin_disabled(tab.busy)" in properties
    assert "busy_reason(tab)" in properties


# --- the alpha lock on a landing ------------------------------------------------


def _locked_doc():
    doc = inker.Document.blank(8, 8)
    layer = doc.stack.active
    layer.pixels[..., :3] = (5, 6, 7)  # hidden colour under alpha 0
    layer.pixels[..., 3] = 0
    layer.pixels[1, 1] = (9, 9, 9, 255)
    doc.set_layer_props(alpha_lock=True)
    doc.history.clear()
    return doc


def test_a_float_landing_on_an_alpha_locked_layer_leaves_transparent_rgb_alone():
    doc = _locked_doc()
    before = doc.stack.active.pixels.copy()
    head = doc.history.head
    stamp = np.zeros((4, 4, 4), dtype=np.uint8)
    stamp[...] = (255, 0, 0, 255)

    assert doc.float_pixels(stamp, (3, 3))
    doc.commit_floating()

    after = doc.stack.active.pixels
    assert np.array_equal(after, before), "hidden colour under alpha 0 was overwritten"
    assert doc.history.head == head, "nothing visible changed, so nothing should be undoable"


def test_a_float_landing_on_an_alpha_locked_layer_still_recolours_what_is_there():
    doc = _locked_doc()
    stamp = np.zeros((2, 2, 4), dtype=np.uint8)
    stamp[...] = (255, 0, 0, 255)

    assert doc.float_pixels(stamp, (0, 0))
    doc.commit_floating()

    after = doc.stack.active.pixels
    assert tuple(after[1, 1]) == (255, 0, 0, 255), "an opaque pixel takes the paste"
    assert int(after[0, 0, 3]) == 0 and tuple(after[0, 0, :3]) == (5, 6, 7)
    assert tuple(after[0, 1, :3]) == (5, 6, 7) and int(after[0, 1, 3]) == 0


def test_an_unlocked_landing_is_unchanged():
    doc = _locked_doc()
    doc.set_layer_props(alpha_lock=False)
    stamp = np.zeros((2, 2, 4), dtype=np.uint8)
    stamp[...] = (255, 0, 0, 255)

    assert doc.float_pixels(stamp, (3, 3))
    doc.commit_floating()

    assert tuple(doc.stack.active.pixels[3, 3]) == (255, 0, 0, 255)


# --- the sheet-merge refusal -------------------------------------------------------


def test_the_no_base_sentence_covers_a_pixel_restyled_sheet_too():
    assert "restyle" in inker_sheet.NO_BASE
    # The sentence the Poser wiring test has always pinned stays in it.
    assert "not opened from a rendered sheet" in inker_sheet.NO_BASE
