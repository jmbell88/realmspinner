"""The 2026-10-03 audit's Medium Inker pane findings, inker-51 .. inker-56 and
inker-62 .. inker-65.

Each test's name is the claim, and each fails against the code as it stood
before the fix. The two manual findings (inker-56, inker-65) are about
``docs/manual/``, which the fixer that wrote these did not own: those two tests
stay red until the chapters are edited, and say which sentence is wrong.
"""

from __future__ import annotations

import ast
import inspect
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import transform
from realmspinner.kernels.pixel.tiles import strip
from realmspinner.studio import widgets
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.ui.panes import bridge as inker_bridge
from realmspinner.studio.modes.inker.ui.panes import colors as inker_colors
from realmspinner.studio.modes.inker.ui.panes import context as inker_context
from realmspinner.studio.modes.inker.ui.panes import gestures as inker_gestures
from realmspinner.studio.modes.inker.ui.panes import picker as inker_picker
from realmspinner.studio.modes.inker.ui.panes import tools as inker_tools
from realmspinner.studio.tasks import Done

REPO = Path(__file__).resolve().parents[3]
PANES = REPO / "src" / "realmspinner" / "studio" / "modes" / "inker" / "ui" / "panes"


class _Prompts:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, prompt) -> None:
        self.asked.append(prompt)


class _Ctx:
    """The little of ``AppCtx`` Inker's ops, tasks and popups touch."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), preview={})
        self.toasts: list[tuple[str, str]] = []
        self.prompts = _Prompts()
        self.submitted: list[tuple[str, object, tuple]] = []
        self.settings = SimpleNamespace(set=lambda *a, **k: None, get=lambda *a, **k: None)

    def toast(self, text: str, level: str = "info", **_: object) -> None:
        self.toasts.append((text, level))

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        self.submitted.append((key, fn, args))
        return True


def _open(ctx: _Ctx, doc=None, title: str = "t.ora") -> inker_state.InkerDoc:
    tab = inker_state.InkerDoc(doc=doc or inker.Document.blank(16, 16), title=title)
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    return tab


# -- inker-51 -----------------------------------------------------------------


def test_the_rename_layer_op_asks_for_a_name_instead_of_naming_a_popup_nobody_answers():
    ctx = _Ctx()
    tab = _open(ctx)
    doc = tab.doc
    doc.add_layer()
    op = inker_ops.get("rename_layer")

    assert inker_ops.run(ctx, op)

    assert ctx.state.inker.pending_dialog == "", (
        "the op parked a dialog name that no pane answers"
    )
    assert len(ctx.prompts.asked) == 1
    prompt = ctx.prompts.asked[0]
    assert prompt.title == "Rename layer"
    assert prompt.value == doc.stack.active.name
    target = doc.stack.active.uid
    prompt.on_accept("Outlines")
    assert next(layer for layer in doc.stack if layer.uid == target).name == "Outlines"


def test_every_dialog_op_names_a_popup_some_pane_answers():
    # Everything in the mode but the registry itself: a name defined as a
    # constant in ``flourish.py`` and matched in a pane is answered.
    answered = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PANES.parent.parent.rglob("*.py")
        if path.name != "ops.py"
    )
    for op in inker_ops.OPS:
        if op.run.__qualname__ != "dialog.<locals>._open":
            continue
        name = next(
            cell.cell_contents
            for cell in op.run.__closure__
            if isinstance(cell.cell_contents, str)
        )
        assert f'"{name}"' in answered, f"{op.name} asks for {name!r} and no pane names it"


# -- inker-52 -----------------------------------------------------------------


def _slice_harness(monkeypatch, *, tick: str):
    """``_slice_options`` with the imgui calls stubbed out and one checkbox
    (``tick``) reporting that the user just switched it on."""
    ctx = _Ctx()
    doc = inker.Document.blank(16, 16)
    doc.ensure_animation()
    tab = _open(ctx, doc)
    entry = doc.add_slice((2, 2, 10, 10), name="hit")
    frame = tab.frame_uid
    assert frame is not None
    assert doc.set_slice_key(entry.uid, frame)

    def checkbox(label, value, **_):
        return (label.startswith(tick), True) if label.startswith(tick) else (False, value)

    imgui = MagicMock(is_item_deactivated_after_edit=lambda: False)
    monkeypatch.setattr(inker_tools, "imgui", imgui)
    monkeypatch.setattr(inker_tools, "widgets", MagicMock())
    monkeypatch.setattr(inker_tools.controls, "checkbox", checkbox)
    monkeypatch.setattr(inker_tools.controls, "input_text", lambda _l, text, *a, **k: (False, text))
    monkeypatch.setattr(inker_tools.controls, "button", lambda *a, **k: False)
    monkeypatch.setattr(inker_tools, "_nineslice_fit_button", lambda *a, **k: None)
    monkeypatch.setattr(inker_tools, "_nineslice_preview", lambda *a, **k: None)
    return ctx, tab, entry, frame


def test_pivot_toggle_on_a_keyed_frame_edits_that_frames_key(monkeypatch):
    ctx, tab, entry, frame = _slice_harness(monkeypatch, tick="Pivot")
    inker_tools._slice_options(ctx, ctx.state.inker, tab, entry)
    assert entry.keys[frame].pivot is not None, "the keyed frame's own pivot never changed"
    assert entry.pivot is None, "the base (and every unkeyed frame) gained the pivot"


def test_nine_slice_toggle_on_a_keyed_frame_edits_that_frames_key(monkeypatch):
    ctx, tab, entry, frame = _slice_harness(monkeypatch, tick="Nine-slice")
    inker_tools._slice_options(ctx, ctx.state.inker, tab, entry)
    assert entry.keys[frame].center is not None
    assert entry.center is None


def test_pivot_toggle_on_an_unkeyed_frame_still_edits_the_base(monkeypatch):
    ctx, tab, entry, frame = _slice_harness(monkeypatch, tick="Pivot")
    assert tab.doc.set_slice_key(entry.uid, frame, clear=True)
    inker_tools._slice_options(ctx, ctx.state.inker, tab, entry)
    assert entry.pivot is not None


# -- inker-53 -----------------------------------------------------------------


def test_a_text_stamp_after_a_tab_switch_is_refused_not_placed_at_the_origin():
    ctx = _Ctx()
    a, b = _open(ctx, title="a"), _open(ctx, title="b")
    state = ctx.state.inker
    state.activate(a.uid)
    state.text_buffer = "your text"
    state.text_at = (12, 9)
    state.text_uid = a.uid
    state.activate(b.uid)  # every tab switch runs clear_drag

    assert inker_mode.stamp_text(ctx, state, b) is False
    assert b.doc.floating is None, "the stamp landed in the other document"


class _PopupStub:
    def __init__(self, *, open_: bool) -> None:
        self.open_ = open_
        self.calls: list[str] = []

    def begin_popup(self, name):
        return self.open_

    def end_popup(self):
        self.calls.append("end")

    def close_current_popup(self):
        self.calls.append("close")

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append(name)


def test_a_text_popup_left_open_across_a_tab_switch_closes_itself(monkeypatch):
    ctx = _Ctx()
    a, b = _open(ctx, title="a"), _open(ctx, title="b")
    state = ctx.state.inker
    state.activate(a.uid)
    state.text_uid = a.uid
    state.activate(b.uid)
    stub = _PopupStub(open_=True)
    controls = MagicMock()
    controls.checkbox.return_value = (False, True)
    controls.button.return_value = False
    widgets_stub = MagicMock()
    widgets_stub.labeled_slider_int.return_value = (False, 1)
    monkeypatch.setattr(inker_gestures, "imgui", stub)
    monkeypatch.setattr(inker_gestures, "controls", controls)
    monkeypatch.setattr(inker_gestures, "widgets", widgets_stub)
    inker_gestures._text_popup(ctx, state, b)
    assert "close" in stub.calls and stub.calls[-1] == "end"
    controls.button.assert_not_called()  # no OK button for a popup that is closed


def test_a_second_canvas_views_popup_miss_does_not_blank_the_text_owner(monkeypatch):
    ctx = _Ctx()
    a = _open(ctx, title="a")
    state = ctx.state.inker
    state.text_uid = a.uid
    monkeypatch.setattr(inker_gestures, "imgui", _PopupStub(open_=False))
    inker_gestures._text_popup(ctx, state, a)
    assert state.text_uid == a.uid


# -- inker-54 -----------------------------------------------------------------


def test_image_size_dialog_does_not_measure_the_lattice_on_the_frame_thread(monkeypatch):
    ctx = _Ctx()
    tab = _open(ctx)
    seen: list[threading.Thread] = []

    def measure(pixels):
        seen.append(threading.current_thread())
        return {"scale": 4, "phase": (0, 0)}

    monkeypatch.setattr(transform, "detect_pixel_grid", measure)
    inker_bridge._measure_pixel_grid(ctx, tab)

    assert seen == [], "the lattice was measured before the frame returned"
    assert len(ctx.submitted) == 1
    key, fn, args = ctx.submitted[0]
    assert key.startswith("inker-grid:")
    # And the answer, once the task lands, is what the Descale row reads.
    inker_mode.on_task_done(ctx, Done(key=key, result=fn(*args)))
    assert ctx.state.preview[f"inker_grid:{tab.uid}"]["scale"] == 4


def test_a_lattice_measured_for_pixels_since_edited_is_dropped(monkeypatch):
    ctx = _Ctx()
    tab = _open(ctx)
    found = {"scale": 4, "phase": (0, 0)}
    monkeypatch.setattr(transform, "detect_pixel_grid", lambda pixels: found)
    inker_bridge._measure_pixel_grid(ctx, tab)
    key, fn, args = ctx.submitted[0]
    result = fn(*args)
    tab.doc.rev += 1  # the user painted while it was being measured
    inker_mode.on_task_done(ctx, Done(key=key, result=result))
    assert f"inker_grid:{tab.uid}" not in ctx.state.preview


# -- inker-55 -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_the_symmetry_axis_field_refuses_a_non_finite_value(monkeypatch, bad):
    ctx = _Ctx()
    tab = _open(ctx)
    state = ctx.state.inker
    state.symmetry_axis = (5.0, 6.0)
    stub = SimpleNamespace(
        set_next_item_width=lambda *a: None,
        is_item_deactivated_after_edit=lambda: False,
        same_line=lambda *a, **k: None,
    )
    monkeypatch.setattr(inker_context, "imgui", stub)
    monkeypatch.setattr(inker_context.controls, "input_float2", lambda *a, **k: (True, [bad, 3.0]))
    monkeypatch.setattr(inker_context.widgets, "disabled_button", lambda *a, **k: False)
    monkeypatch.setattr(inker_context.widgets, "muted", lambda *a, **k: None)
    inker_context._symmetry_axis(ctx, state, tab)
    assert state.symmetry_axis == (5.0, 6.0)


# -- inker-56 and inker-65 (the manual) ----------------------------------------


def _chapter(name: str) -> str:
    text = (REPO / "docs" / "manual" / name).read_text(encoding="utf-8")
    return " ".join(text.split())


def test_manual_selection_modifier_chords_match_the_registry():
    # What the registry binds: add Shift, subtract Alt+Shift, intersect Ctrl+Shift.
    held = {
        "selection_add": "Shift",
        "selection_subtract": "Alt+Shift",
        "selection_intersect": "Ctrl+Shift",
    }
    for action, chord in held.items():
        assert inker_ops.action_active(action, chord, "Selection"), (action, chord)
    assert not inker_ops.action_active("selection_subtract", "Alt", "Selection")

    chapter = _chapter("28-inker.md")
    start = chapter.index("## Selections and transform")
    opening = chapter[start : start + 400]
    assert "**Alt** to subtract" not in opening, (
        "chapter 28's Selections section says Alt alone subtracts; Alt alone replaces"
    )
    assert "Alt+Shift" in opening and "Ctrl+Shift" in opening


def test_inker_manual_does_not_describe_controls_that_moved():
    ch28, ch29 = _chapter("28-inker.md"), _chapter("29-inker-animation.md")
    # There is no "selection section": All/None/Invert/Feather/Crop are Select-menu rows.
    assert "the **selection** section" not in ch28
    # Free transform is the Edit menu and Ctrl+T; the tool-options button was removed.
    assert "or the button in the tool options" not in ch28
    # The context bar gains a combine-mode row whenever a mask exists.
    assert "exactly one row above the canvas" not in ch28
    # The exports left the timeline's second row on 2026-09-05.
    assert "the three exports" not in ch29
    # The Cels toggle is in Layer properties, not at the top of a layers panel.
    assert "toggle at the top of the layers panel" not in ch29
    # The range menu writes a PNG sequence as well.
    part = ch29[ch29.index("## Exporting part of a clip") :][:500]
    assert "PNG sequence" in part


# -- inker-62 -----------------------------------------------------------------


def test_the_picker_refuses_to_edit_a_palette_slot_while_the_tab_is_busy():
    ctx = _Ctx()
    state = ctx.state.inker
    recoloured: list = []

    class _Doc:
        palette = [(0, 0, 0, 255), (255, 0, 0, 255)]
        is_indexed = True

        def recolour_slot(self, index, colour):
            recoloured.append((index, colour))
            return True

    tab = SimpleNamespace(doc=_Doc(), busy=True)
    inker_picker.write(ctx, state, tab, 1, (0, 128, 255, 255))
    assert recoloured == [], "a palette swap and an undo step were written mid-save"
    tab.busy = False
    inker_picker.write(ctx, state, tab, 1, (0, 128, 255, 255))
    assert recoloured == [(1, (0, 128, 255, 255))]


def test_a_free_colour_is_still_picked_while_the_tab_is_busy():
    """Only the slot path writes the document; the brush colour is session state."""
    ctx = _Ctx()
    state = ctx.state.inker
    tab = SimpleNamespace(doc=SimpleNamespace(palette=[], is_indexed=False), busy=True)
    inker_picker.write(ctx, state, tab, None, (1, 2, 3, 255))
    assert state.fg == (1, 2, 3, 255)


# -- inker-63 -----------------------------------------------------------------


def _tilemap_range_tab(ctx: _Ctx):
    doc = inker.Document.blank(8, 8)
    anim = doc.ensure_animation()
    doc.stack[0].pixels[...] = 255
    tile = np.zeros((4, 4, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 3] = 255, 255
    slot = doc.add_tileset(strip(np.stack([np.zeros((4, 4, 4), np.uint8), tile])))
    layer = doc.add_tilemap_layer(slot.uid, name="Ground")
    assert doc.place_tiles(layer.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.set_active_layer(0)
    tab = _open(ctx, doc)
    tab.range_sel = (0, 1, anim.current, anim.current)
    return tab


def test_apply_to_range_over_a_tilemap_track_toasts_instead_of_raising(monkeypatch):
    ctx = _Ctx()
    tab = _tilemap_range_tab(ctx)
    state = ctx.state.inker
    state.filter_name = "invert"
    assert tab.doc.begin_filter() is not None
    state.filter_uid = tab.uid
    stub = MagicMock()
    monkeypatch.setattr(inker_bridge, "imgui", stub)
    monkeypatch.setattr(inker_bridge, "widgets", MagicMock())
    monkeypatch.setattr(inker_bridge.controls, "button", lambda *a, **k: True)

    inker_bridge._apply_to_range(ctx, tab)  # raised ValueError before the fix

    assert ctx.toasts and ctx.toasts[-1][1] == "warn"
    assert "tilemap" in ctx.toasts[-1][0]
    assert state.filter_uid == tab.uid, "the filter session was thrown away by a refusal"
    assert tab.doc._filter is not None
    stub.close_current_popup.assert_not_called()


# -- inker-64 -----------------------------------------------------------------


def _busy_gated_functions():
    """Every function in the four panes whose body greys a block on ``.busy``."""
    for module in (inker_colors, inker_tools, inker_bridge, inker_picker):
        tree = ast.parse(inspect.getsource(module))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            greys = [
                call
                for call in ast.walk(func)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "begin_disabled"
                and any(
                    isinstance(node, ast.Attribute) and node.attr == "busy"
                    for arg in call.args
                    for node in ast.walk(arg)
                )
            ]
            if greys:
                yield module.__name__.rsplit(".", 1)[-1], func


def test_every_busy_greyed_inker_control_names_the_busy_reason():
    found = list(_busy_gated_functions())
    assert found, "the walk found no busy-greyed block at all; it is looking in the wrong place"
    for module, func in found:
        names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Attribute | ast.Name)
        }
        assert names & {"DOCUMENT_SAVING_WHY", "busy_reason", "_busy_gate"}, (
            f"{module}.{func.name} greys controls on tab.busy and says nothing about why"
        )


def test_the_busy_reason_is_said_on_hover_only_while_the_tab_is_busy(monkeypatch):
    said: list[str] = []
    stub = SimpleNamespace(
        is_item_hovered=lambda *a: True,
        set_tooltip=said.append,
        HoveredFlags_=SimpleNamespace(allow_when_disabled=SimpleNamespace(value=0)),
    )
    monkeypatch.setattr(inker_colors, "imgui", stub)
    inker_colors.busy_reason(SimpleNamespace(busy=False))
    assert said == []
    inker_colors.busy_reason(SimpleNamespace(busy=True))
    assert said == [widgets.DOCUMENT_SAVING_WHY]
