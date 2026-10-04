"""The 2026-10-03 audit's Low findings for Inker, batch 2 (inker2).

Each test's name is the claim. Where a finding is a missing agreement test
(inker-84) or a doc fix (inker-91, inker-98) the test is the fix or the gate,
and says so.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import brush, dither, indexed
from realmspinner.studio.modes.inker import export as inker_export
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import state as inker_state

RED = (255, 0, 0, 255)
REPO = Path(__file__).resolve().parents[3]


# --- inker-71: a blur or smudge dab leaves the hidden RGB of transparent pixels alone ---------


def _hidden_rgb_layer():
    """Opaque red on the left; on the right, fully transparent pixels that still
    carry the colour they had before they were erased."""
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[:, :16] = RED
    pixels[:, 16:] = (9, 8, 7, 0)
    return pixels


@pytest.mark.parametrize("mode", ["blur", "smudge"])
def test_a_zero_strength_blur_dab_leaves_every_byte_of_a_transparent_region_alone(mode):
    pixels = _hidden_rgb_layer()
    original = pixels.copy()
    stroke = brush.StrokeState(
        layer_uid=1,
        size=(32, 32),
        before=pixels.copy(),
        colour=RED,
        diameter=16,
        hardness=1.0,
        mode=mode,
        strength=0.0,
    )

    stroke.begin((24, 16), pixels)

    assert np.array_equal(pixels, original), "the dab rewrote bytes it never meant to touch"


# --- inker-73: a grouped indexed conversion uses both of two identical slots --------------------


def _two_greys():
    plane = np.zeros((2, 1, 4), dtype=np.uint8)
    plane[..., 3] = 255
    plane[0, 0, :3] = 10
    plane[1, 0, :3] = 240
    return plane


def test_a_grouped_indexed_conversion_uses_both_of_two_identical_palette_slots():
    brown = (120, 60, 20, 255)
    palette = [(0, 0, 0, 0), brown, brown]
    plane = _two_greys()

    table = dither.grouped_index_table([plane], palette)
    with_table = dither.convert_indices(plane, palette, "grouped", table=table)
    without = dither.convert_indices(plane, palette, "grouped")

    assert set(np.unique(with_table).tolist()) == {1, 2}
    assert set(np.unique(without).tolist()) == {1, 2}


def test_a_plain_two_tuple_table_still_converts_indices():
    """A caller's own ``(keys, targets)`` has no ``.slots`` and keeps the
    by-colour answer rather than failing."""
    palette = [(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255)]
    plane = _two_greys()
    keys, targets = dither.grouped_index_table([plane], palette)

    out = dither.convert_indices(plane, palette, "grouped", table=(keys, targets))

    assert out[:, 0].tolist() == [1, 2]


# --- inker-75: a zero-alpha text stamp declines -----------------------------------------------


def test_text_stamp_with_a_zero_alpha_colour_declines_rather_than_returning_a_blank_array():
    from realmspinner.kernels.pixel.textstamp import text_stamp
    from realmspinner.studio import fonts

    font = str(fonts.FONT_DIR / "Inter-Regular.ttf")
    assert text_stamp("Hi", font, 24, (255, 0, 0, 255)) is not None, "fixture: the font works"

    assert text_stamp("Hi", font, 24, (255, 0, 0, 0)) is None


# --- inker-80: a 256-colour indexed GIF export says it is not writing the table verbatim ----------


class _Ctx:
    def __init__(self, state):
        self.state = SimpleNamespace(inker=state)
        self.toasts: list[tuple[str, str]] = []
        self.run = None

    def toast(self, message, kind="info", **_kw):
        self.toasts.append((message, kind))

    def submit(self, key, run):
        self.run = run
        return True


def _indexed_clip(colours):
    doc = inker.Document.blank(4, 4)
    doc.write_colour((0, 0, 2, 2), RED, np.ones((2, 2), dtype=np.float32))
    doc.add_frame(link=True)
    assert doc.convert_to_indexed(colours) is True
    tab = inker_state.InkerDoc(doc=doc, title="walk.ora")
    state = inker_state.InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _export_gif(monkeypatch, tmp_path, colours):
    from realmspinner.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "clip.gif")
    ctx, state, tab = _indexed_clip(colours)
    inker_mode.export_gif(ctx, tab)
    for _ in range(50):
        inker_mode.pump_export(ctx)
        if state.export is None:
            break
    assert ctx.run is not None, "fixture: the export reached its task"
    return ctx, tab, ctx.run()


def test_a_256_colour_indexed_gif_export_says_it_is_not_writing_the_table_verbatim(
    monkeypatch, tmp_path
):
    full = [(i, 255 - i, (i * 7) % 256, 255) for i in range(256)]
    ctx, tab, result = _export_gif(monkeypatch, tmp_path, full)

    notes = result.get("notes") or []
    assert notes and "256" in notes[0], result

    inker_mode.on_task_done(ctx, SimpleNamespace(key=f"inker-export:{tab.uid}", result=result))
    text, level = ctx.toasts[-1]
    assert "Exported to" in text and "own palette" in text and level == "warn", ctx.toasts


def test_a_small_indexed_gif_export_stays_a_plain_toast(monkeypatch, tmp_path):
    ctx, tab, result = _export_gif(monkeypatch, tmp_path, [(0, 0, 0, 255), RED])

    assert not result.get("notes")
    inker_mode.on_task_done(ctx, SimpleNamespace(key=f"inker-export:{tab.uid}", result=result))
    assert ctx.toasts[-1][1] == "info"


# --- inker-82: the Colours slider reaches what the recipe format allows --------------------------


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_colours_slider_reaches_what_the_recipe_format_allows(ui, monkeypatch):
    import dataclasses

    from realmspinner.kernels.pixel.flourish import bake as B
    from realmspinner.kernels.pixel.flourish import presets
    from realmspinner.kernels.pixel.flourish import recipe as flourish_recipe
    from realmspinner.studio import probe
    from realmspinner.studio.modes.inker.ui.panes import flourish as pane

    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=inker_state.InkerState(), manual=None),
        submitted=[],
        toast=lambda *a, **k: None,
        busy=lambda key: False,
        progress=lambda key: None,
        submit=lambda *a, **k: True,
        tasks=SimpleNamespace(set_progress=lambda *a, **k: None),
    )
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(
        presets.load("sword_impact"), width=32, height=32, supersample=2, mode="pixel"
    )
    tab.doc.insert_flourish(B.bake(rec))

    seen: dict[str, tuple[int, int]] = {}
    real = pane.controls.slider_int

    def _spy(label, value, lo, hi, *a, **k):
        seen[label] = (lo, hi)
        return real(label, value, lo, hi, *a, **k)

    monkeypatch.setattr(pane.controls, "slider_int", _spy)
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        pane.draw_inspector(ctx, tab)
    finally:
        ui.end()
        ui.end_frame()

    colours = next(v for k, v in seen.items() if k.startswith("##colours"))
    assert colours == (2, flourish_recipe.MAX_COLORS)


# --- inker-84: the engine snippet's atlas ceiling is the sheet kernel's -------------------------


def test_the_engine_snippets_atlas_ceiling_matches_the_sheet_kernels():
    from realmspinner.kernels import sheet
    from realmspinner.kernels.pixel.flourish import engines

    assert engines._MAX_ATLAS_PX == sheet.MAX_ATLAS_PX


# --- inker-86: a failed palette pick does not clear a running save lock ------------------------


@pytest.mark.parametrize("head", ["inker-index", "inker-palimg", "inker-tileset-import"])
def test_a_failed_palette_pick_does_not_clear_a_running_save_lock(head):
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), title="a.ora")
    state.add(tab)
    tab.saving = True
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    inker_mode.on_task_failed(ctx, SimpleNamespace(key=f"{head}:{tab.uid}"))

    assert tab.saving is True


@pytest.mark.parametrize(
    "head",
    ["inker-save", "inker-saveas", "inker-export", "inker-send", "inker-revert", "inker-convert"],
)
def test_a_failed_save_still_clears_the_lock(head):
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), title="a.ora")
    state.add(tab)
    tab.saving = True
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    inker_mode.on_task_failed(ctx, SimpleNamespace(key=f"{head}:{tab.uid}"))

    assert tab.saving is False


# --- inker-87: the Export doors name playback when playback greys them ------------------------


def test_the_export_doors_name_playback_when_playback_is_what_greys_them():
    tab = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), title="a.ora")
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    tab.playing = True
    sheet = next(d for d in inker_export.doors() if d.key == "sheet")

    enabled, reason = inker_export.door_state(sheet, tab)

    assert not enabled
    assert reason == inker_export.PLAYING_WHY and "write" not in reason.lower()


def test_a_save_still_gets_the_write_sentence_even_while_playing():
    tab = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), title="a.ora")
    tab.playing = True
    tab.saving = True
    sheet = next(d for d in inker_export.doors() if d.key == "sheet")

    assert inker_export.door_state(sheet, tab) == (False, inker_export.BUSY_WHY)


# --- inker-88: the Merge re-render lists sheets on the task thread ----------------------------


def test_merge_rerender_lists_sheets_on_the_task_thread(monkeypatch):
    import threading

    from realmspinner.kernels.pixel.sheetin import document_from_sheet
    from realmspinner.studio.modes.inker import ops as inker_ops

    cell = np.zeros((8, 8, 4), dtype=np.uint8)
    cell[..., 3] = 255
    atlas = np.concatenate([cell, cell], axis=1)
    cells = [{"x": 0, "y": 0, "w": 8, "h": 8}, {"x": 8, "y": 0, "w": 8, "h": 8}]
    anim = {"tags": [{"name": "walk_front", "start": 0, "end": 1, "loop": True}], "frames": []}
    doc = document_from_sheet(atlas, cells, anim, source={"job": "J", "sheet": "S1"})
    tab = inker_state.InkerDoc(doc=doc, title="walk.ora")
    state = inker_state.InkerState()
    state.add(tab)

    seen: list[str] = []
    submitted: list = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        svc=object(),
        toast=lambda *a, **k: None,
        submit=lambda key, fn, *a, **k: submitted.append(fn) or True,
    )

    def _newest(svc, job, sheet):
        seen.append(threading.current_thread().name)
        return ""

    monkeypatch.setattr(inker_mode, "newest_sheet_after", _newest)

    inker_ops._sheet_merge(ctx, tab)

    assert seen == [], "the sheet listing ran in the click, on the frame thread"
    assert len(submitted) == 1
    submitted[0]()
    assert seen, "the listing moved into the submitted task"


# --- inker-90: Alt+0 does not set the opacity --------------------------------------------------


def test_alt_zero_does_not_set_the_opacity():
    from types import MethodType

    import pygame

    from realmspinner.studio import state as state_mod

    tab = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="t1", title="t")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)
    state.opacity = 0.4

    inker_mode.handle_key(
        ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, mod=pygame.KMOD_ALT)
    )

    assert state.opacity == 0.4


# --- inker-91: the manual names only preview controls the pane has --------------------------------


def test_the_manual_names_only_preview_controls_the_preview_pane_has():
    chapter = (REPO / "docs" / "manual" / "06-animating.md").read_text(encoding="utf-8")
    pane = (
        REPO / "src" / "realmspinner" / "studio" / "modes" / "inker" / "ui" / "panes" / "preview.py"
    ).read_text(encoding="utf-8")
    sentence = re.search(r"Playback \*speed\*.*?preview pane", chapter, re.S)
    assert sentence, "the sentence about where playback settings live is gone"
    assert "once" not in sentence.group(0).lower() or "once" in pane.lower()


# --- inker-92: no two colour harmonies produce the same offsets ----------------------------------


def test_no_two_colour_harmonies_produce_the_same_offsets():
    seen: dict[tuple[float, ...], str] = {}
    for name, offsets in indexed.HARMONIES.items():
        assert offsets not in seen, f"{name} and {seen.get(offsets)} are the same harmony"
        seen[offsets] = name


# --- inker-95: choosing "From selection" in a walk part combo assigns ----------------------------


def _walk_row_scene():
    from realmspinner.studio.modes.inker import walk as inker_walk

    tab = inker_state.InkerDoc(doc=inker.Document.blank(64, 64))
    tab.doc.stack.active.pixels[18:38, 28:36] = (200, 80, 80, 255)
    state = inker_state.InkerState()
    state.docs.append(tab)
    state.active_uid = tab.uid
    toasts: list = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state, manual=None, preview={}),
        toast=lambda text, level="info", **_: toasts.append((text, level)),
        viewer=None,
    )
    inker_walk.open_session(ctx, tab)
    return ctx, tab, toasts


def test_choosing_from_selection_in_a_walk_part_combo_assigns_or_is_not_offered(ui, monkeypatch):
    from realmspinner.studio.modes.inker.ui.panes import walk as pane

    ctx, tab, toasts = _walk_row_scene()
    session = ctx.state.inker.walk
    tab.doc.select_all()
    monkeypatch.setattr(pane.widgets, "combo", lambda *a, **k: "selection")
    monkeypatch.setattr(pane.widgets, "disabled_button", lambda *a, **k: False)

    ui.new_frame()
    ui.begin("host")
    try:
        pane._part_row(ctx, tab, session, "torso", pane._layer_options(tab))
    finally:
        ui.end()
        ui.end_frame()

    assert session.assigned_from.get("torso") == "selection"
    assert session.rig.parts["torso"].assigned


def test_choosing_from_selection_with_no_selection_says_why(ui, monkeypatch):
    from realmspinner.studio.modes.inker.ui.panes import walk as pane

    ctx, tab, toasts = _walk_row_scene()
    session = ctx.state.inker.walk
    monkeypatch.setattr(pane.widgets, "combo", lambda *a, **k: "selection")
    monkeypatch.setattr(pane.widgets, "disabled_button", lambda *a, **k: False)

    ui.new_frame()
    ui.begin("host")
    try:
        pane._part_row(ctx, tab, session, "torso", pane._layer_options(tab))
    finally:
        ui.end()
        ui.end_frame()

    assert "torso" not in session.assigned_from
    assert toasts, "a pick that cannot run must say why"


# --- inker-96: release_dropped frees the current-layer onion variants too ------------------------


class _Texture:
    def __init__(self):
        self.size = (4, 4)
        self.released = False

    def release(self):
        self.released = True


def test_release_dropped_frees_the_current_layer_onion_variants_too():
    from realmspinner.studio.modes.inker.ui.panes import textures

    doc = SimpleNamespace(take_dropped_frames=lambda: [7])
    tab = SimpleNamespace(uid="t1", doc=doc)
    preview = {}
    ctx = SimpleNamespace(viewer=object(), state=SimpleNamespace(preview=preview))
    plain = textures._slot("t1", "frame7")
    layer = textures._slot("t1", "frame7t3")
    other = textures._slot("t1", "frame70")  # a different frame, must survive
    held = {key: _Texture() for key in (plain, layer, other)}
    for key, texture in held.items():
        preview[key] = texture
        preview[f"{key}:rev"] = 1
        textures._frame_lru(ctx, "t1")[key] = None
        textures._frame_touched(ctx, "t1")[key] = 1

    textures.release_dropped(ctx, tab)

    assert held[plain].released and held[layer].released
    assert not held[other].released and other in preview
    assert layer not in preview and f"{layer}:rev" not in preview
    assert layer not in textures._frame_lru(ctx, "t1")
    assert layer not in textures._frame_touched(ctx, "t1")
    assert other in textures._frame_lru(ctx, "t1")


# --- inker-97: submit_inpaint resizes and encodes off the frame thread ---------------------------


def test_submit_inpaint_does_the_resize_and_encode_off_the_frame_thread(monkeypatch):
    from realmspinner.kernels.pixel import inpaint
    from realmspinner.service import jobs as svc_jobs
    from realmspinner.studio.modes.inker.ui.panes import bridge

    tab = inker_state.InkerDoc(doc=inker.Document.blank(64, 64))
    tab.doc.select_all()
    state = inker_state.InkerState()
    state.add(tab)
    submitted: list = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        svc=object(),
        toast=lambda *a, **k: None,
        submit=lambda key, fn, *a, **k: submitted.append(fn) or True,
    )
    encoded: list[str] = []
    real_png = inpaint._png
    monkeypatch.setattr(inpaint, "_png", lambda image: encoded.append("png") or real_png(image))
    sent: dict = {}
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: sent.update(kw) or {"id": "j"})

    assert bridge.submit_inpaint(ctx, tab, "a red door", 0.5) is True

    assert encoded == [], "the resize and PNG encodes ran inside the click"
    submitted[0]()
    assert encoded == ["png", "png"]
    assert sent["reference"][:4] == b"\x89PNG" and sent["mask"][:4] == b"\x89PNG"


# --- inker-98: the geometry docstrings name the tilemap ops the document actually models --------


def test_geometry_docstrings_name_the_tilemap_ops_the_document_actually_models():
    from realmspinner.kernels.pixel import _doc_geometry

    module = _doc_geometry.__doc__
    assert "exactly one of the five" not in module
    assert "no such algebra is written yet" not in module
    assert "flip" in module.split("A tilemap layer survives")[1].split("**")[0]
    resize = _doc_geometry.GeometryOps.resize_canvas.__doc__
    assert "the one geometry op a tilemap layer survives" not in resize.lower().replace("\n", " ")
    # And the doc really does model them, which is what the sentence claims.
    assert "A tilemap layer comes with it" in _doc_geometry.GeometryOps.flip.__doc__
    assert "A tilemap layer comes with them" in _doc_geometry.GeometryOps.rotate90.__doc__
    assert inspect.getsource(_doc_geometry).count("_refuse_tilemaps(") >= 3
