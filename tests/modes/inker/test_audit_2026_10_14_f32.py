"""The 2026-10-14 audit, fixer F32: the Inker panes and the walk-cycle session.

Findings inker-panes-11 (a bad hex entry says nothing), -03 (the tile-size popup
has no ceiling), -09 (the tile panel's button is named differently from the
manual), -06 (the timeline submits every cell), -07 (the eyes are greyed with no
reason), -08 (the FPS box is in no chapter), -10 (a dead nib table), and
inker-mode-14 / -15 (a joint dragged off the canvas, a failing walk preview).
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
from realmspinner.kernels.pixel import walk
from realmspinner.kernels.pixel.tiles import strip
from realmspinner.studio import probe, widgets
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker import walk as inker_walk
from realmspinner.studio.modes.inker.ui.panes import picker as inker_picker
from realmspinner.studio.modes.inker.ui.panes import tiles as inker_tiles
from realmspinner.studio.modes.inker.ui.panes import timeline as tl
from realmspinner.studio.modes.inker.ui.panes import tools as inker_tools
from realmspinner.studio.state import AppState

ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "docs" / "manual"


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- inker-panes-11 -----------------------------------------------------------


def test_a_bad_hex_entry_is_reported(monkeypatch):
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, level="info", **_: toasts.append((text, level)))
    monkeypatch.setattr(inker_picker.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(inker_picker.imgui, "set_next_item_width", lambda *a, **k: None)
    monkeypatch.setattr(
        inker_picker.controls, "input_text", lambda *a, **k: (True, "not a colour")
    )
    written: list = []
    monkeypatch.setattr(inker_picker, "write", lambda *a: written.append(a))

    inker_picker._hex(ctx, None, None, None, (1, 2, 3, 255))

    assert written == []
    assert len(toasts) == 1 and toasts[0][1] in ("warn", "error"), toasts
    assert "not a colour" in toasts[0][0] or "hex" in toasts[0][0].lower()


# --- inker-panes-03 -----------------------------------------------------------


def _tile_tab(size: int):
    doc = inker.Document.blank(size, size)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="t")
    state = inker_state.InkerState()
    state.add(tab)
    return state, tab


def test_convert_to_tilemap_refuses_a_grid_with_too_many_cells():
    state, tab = _tile_tab(1024)
    enabled, reason = inker_tiles.convert_gate(state, tab, 1, 1)
    assert enabled is False
    assert "tile" in reason and str(inker_tiles.MAX_TILE_CELLS) in reason.replace(",", "")
    # An ordinary cut is untouched, and so is asking with no size at all.
    assert inker_tiles.convert_gate(state, tab, 16, 16) == (True, "")
    assert inker_tiles.convert_gate(state, tab) == (True, "")


# --- inker-panes-09 -----------------------------------------------------------


def test_the_tiles_panel_button_matches_the_manual_s_name(ui):
    chapter = (MANUAL / "25-inker.md").read_text(encoding="utf-8")
    named = re.findall(r"\*\*(New tilemap layer|Add a tilemap layer)\*\*", chapter)
    assert named, "the chapter no longer names the button"
    state, tab = _tile_tab(32)
    tab.doc.add_tileset(strip(np.zeros((2, 8, 8, 4), dtype=np.uint8)))
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), toast=lambda *a, **k: None)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    inker_tiles._verbs(ctx, state, tab)
    ui.end()
    ui.end_frame()

    drawn = [c.label for c in probe.FRAME_CONTROLS]
    assert any(named[0] in label for label in drawn), (named, drawn)


# --- inker-panes-06 -----------------------------------------------------------


def _long_clip(frames: int, layers: int):
    doc = inker.Document.blank(16, 16)
    for _ in range(layers - 1):
        doc.add_layer()
    for _ in range(frames - 1):
        doc.add_frame()
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="x")
    app = AppState()
    app.inker = inker_state.InkerState()
    app.inker.add(tab)
    return SimpleNamespace(state=app, viewer=None), tab


def _draw_grid(ui, ctx, tab, *, width=900.0):
    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((width, 300.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    tl._grid(ctx, tab)
    ui.end()
    ui.end_frame()
    return list(probe.FRAME_CONTROLS)


def test_the_timeline_submits_only_the_cells_in_view(ui):
    frames, layers = 300, 6
    ctx, tab = _long_clip(frames, layers)
    controls = _draw_grid(ui, ctx, tab)
    # Cells are the square buttons; the header's number buttons are one row.
    cells = [c for c in controls if c.kind == "button" and c.label in ("", "*", "=")]
    assert cells, [c.label for c in controls][:10]
    assert len(cells) < frames * layers // 3, len(cells)
    # And a row still reaches its first column, so the marquee has geometry.
    assert any(c.label in ("*", "") for c in cells)


def test_visible_columns_is_the_window_plus_one_either_side():
    first, last = tl.visible_columns(x0=100.0, pitch=22.0, lo=0.0, hi=400.0, count=300)
    assert first == 0 and 13 <= last <= 16
    first, last = tl.visible_columns(x0=-2200.0, pitch=22.0, lo=0.0, hi=400.0, count=300)
    assert 98 <= first <= 101 and 117 <= last <= 123
    # Never outside the clip, never empty.
    assert tl.visible_columns(x0=-99999.0, pitch=22.0, lo=0.0, hi=400.0, count=5) == (4, 4)
    assert tl.visible_columns(x0=0.0, pitch=22.0, lo=0.0, hi=400.0, count=0) == (0, -1)


# --- inker-panes-07 -----------------------------------------------------------


def test_the_layer_eye_says_why_it_is_greyed_while_the_clip_plays(ui):
    ctx, tab = _long_clip(3, 2)
    tab.playing = True  # busy is "saving or playing"
    controls = _draw_grid(ui, ctx, tab)
    eyes = [
        c
        for c in controls
        if c.kind == "icon_button" and c.label.endswith(("##visible", "##alleyes", "##alllocks"))
    ]
    assert len(eyes) >= 4, [(c.label, c.kind) for c in controls][:8]
    for eye in eyes:
        assert eye.enabled is False, eye.label
        assert eye.reason == widgets.DOCUMENT_SAVING_WHY, eye.label


def test_a_group_eye_says_why_it_is_greyed_while_the_clip_plays():
    # The group header's eye is drawn only when a group exists; the shape of the
    # call is what is asserted, against the source, because building a group
    # in a probe frame is the same draw the test above already does.
    source = inspect.getsource(tl._group_row)
    assert "begin_disabled(tab.busy)" not in source
    assert "_busy_gate(tab)" in source


# --- inker-panes-08 -----------------------------------------------------------


def test_the_manual_names_the_fps_box():
    chapter = (MANUAL / "26-inker-animation.md").read_text(encoding="utf-8")
    assert "**FPS**" in chapter
    assert "without anything having to be a frame rate" not in chapter


# --- inker-panes-10 -----------------------------------------------------------


def test_no_pane_keeps_an_unused_nib_table():
    assert not hasattr(inker_tools, "NIB_LABELS")


# --- inker-mode-14 ------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), manual=None, preview={})
        self.toasts: list = []
        self.submitted: list = []

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append(key)
        fn(*args, **kwargs)
        return True


def _scene(size=(64, 64)):
    ctx = _Ctx()
    doc = inker.Document.blank(*size)
    doc.stack.active.pixels[18:38, 28:36] = (200, 80, 80, 255)
    tab = inker_state.InkerDoc(doc=doc)
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    assert inker_walk.open_session(ctx, tab)
    return ctx, tab


def test_a_joint_dragged_far_off_the_canvas_is_clamped_to_it():
    ctx, tab = _scene()
    inker_walk.set_joint(ctx, tab, "near_wrist", (5000.0, -9000.0))
    x, y = ctx.state.inker.walk.rig.joints["near_wrist"]
    margin = inker_walk.JOINT_MARGIN
    assert -margin <= x <= 64 + margin and -margin <= y <= 64 + margin, (x, y)
    inker_walk.set_ground(ctx, tab, 1e7)
    assert ctx.state.inker.walk.rig.ground_y <= 64 + margin
    inker_walk.set_ground(ctx, tab, -1e7)
    assert ctx.state.inker.walk.rig.ground_y >= -margin


def test_a_joint_inside_the_canvas_is_left_where_it_was_put():
    ctx, tab = _scene()
    inker_walk.set_joint(ctx, tab, "near_hip", (30.5, 40.25))
    assert ctx.state.inker.walk.rig.joints["near_hip"] == (30.5, 40.25)


# --- inker-mode-15 ------------------------------------------------------------


def _ready_session(ctx, tab, monkeypatch):
    session = ctx.state.inker.walk
    monkeypatch.setattr(inker_walk, "ready", lambda s: True)
    return session


def test_a_failed_walk_preview_is_not_resubmitted_for_the_same_rig(monkeypatch):
    ctx, tab = _scene()
    session = _ready_session(ctx, tab, monkeypatch)
    calls: list[int] = []

    def boom(*_a, **_k):
        calls.append(1)
        raise RuntimeError("the render fell over")

    monkeypatch.setattr(inker_walk.walk, "frames", boom)
    for _ in range(5):
        inker_walk.frames(session, ctx)
    assert len(calls) == 1, "every frame resubmitted a render that cannot succeed"
    assert len(ctx.toasts) == 1 and ctx.toasts[0][1] == "error", ctx.toasts

    # Moving the rig is a new question, and it is asked again.
    session.rig = walk.set_joint(session.rig, "near_hip", (30.0, 40.0))
    inker_walk.frames(session, ctx)
    assert len(calls) == 2
