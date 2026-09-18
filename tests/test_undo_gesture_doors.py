"""One gesture, one undo step -- at every slider and drag door that writes history.

The 2026-09-02 review's first theme: a slider reports a change on every frame
the pointer moves, and a door that pushes a step per report turns one second of
dragging into sixty steps. With ``UNDO_MAX_DEPTH = 64`` that evicts every
earlier edit in the document, so dragging Tempo once cost the user their whole
session's history. ``controls.fold_undo`` is the fix -- ``UndoStack.mark`` on
activation, ``collapse_since`` on release -- and this file is what pins it to
each door it was applied to.

Three layers, cheapest first: the helper alone against a bare stack; the panes
drawn in a real imgui frame with the item state scripted, so the door's own
code runs and the document's own history is counted; and, for the doors that a
headless frame cannot open -- a context-menu popup, or a panel drawn only in
ordinary tab flow -- a source check that the fold sits between the field and
the write. The 2026-09-04 audit added six rows to that last layer: Clay's
material sliders, Plotter's layer and object opacity and tint, and Sirens's
channel Pan had no ``fold_undo`` call in their files at all.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from imgui_bundle import imgui
from test_sirens_mode import FakeCtx
from test_sirens_panes_smoke import _loaded, _no_device  # noqa: F401
from test_sirens_panes_smoke import frames as frames  # noqa: F401, PLC0414

from warlock.core import undo
from warlock.studio import controls, sirens_mode, widgets
from warlock.studio.modes.clay.ui.panes import outliner as clay_outliner
from warlock.studio.modes.clay.ui.panes import props as clay_props
from warlock.studio.panes import (
    inker_colors,
    inker_picker,
    inker_timeline,
    mason_props,
    packwright_settings,
    packwright_sources,
    plotter_layers,
    plotter_tileset_editor,
    sirens_effects,
    sirens_instruments,
    sirens_orders,
    sirens_patterns,
    sirens_transport,
)


@dataclass
class _Step(undo.Edit):
    def undo(self, doc: Any) -> None:
        pass

    def redo(self, doc: Any) -> None:
        pass


class _Item:
    """Scripted item state: which frame the drag begins and which it ends.

    imgui has one active item, so the state is reported only while the
    scripted field is the last one drawn (``label``, set by ``_scripted_slider``;
    every field when ``None``).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, begin: int, end: int) -> None:
        self.frame = 0
        self.begin, self.end = begin, end
        self.label: str | None = None
        self.last: str | None = None
        monkeypatch.setattr(imgui, "is_item_activated", lambda: self._at(self.begin))
        monkeypatch.setattr(imgui, "is_item_deactivated", lambda: self._at(self.end))
        # A test that fails mid-gesture must not leak it into the next.
        monkeypatch.setattr(controls, "_gesture", None)

    def _at(self, frame: int) -> bool:
        return (self.label is None or self.last == self.label) and self.frame == frame

    def dragging(self) -> bool:
        return self.begin <= self.frame < self.end


# --- 1. the helper -------------------------------------------------------------


def test_a_drag_of_seventy_frames_is_one_step_and_evicts_nothing(monkeypatch):
    """Longer than the depth cap on purpose: the eviction is deferred while the
    gesture is open, so the step before the drag survives it."""
    stack = undo.UndoStack()
    earlier = _Step()
    stack.push(earlier)
    item = _Item(monkeypatch, begin=1, end=71)
    for item.frame in range(1, 72):
        controls.fold_undo(stack)
        if item.dragging():
            stack.push(_Step())
    assert len(stack) == 2
    assert stack._done[0] is earlier
    assert isinstance(stack.top, undo.CompoundEdit)
    assert controls._gesture is None


def test_a_drag_that_moved_once_stays_a_plain_step(monkeypatch):
    stack = undo.UndoStack()
    item = _Item(monkeypatch, begin=1, end=2)
    for item.frame in (1, 2):
        controls.fold_undo(stack)
        if item.dragging():
            stack.push(_Step())
    assert len(stack) == 1
    assert not isinstance(stack.top, undo.CompoundEdit)


def test_a_field_without_history_folds_nothing(monkeypatch):
    item = _Item(monkeypatch, begin=1, end=2)
    for item.frame in (1, 2):
        controls.fold_undo(None)
    assert controls._gesture is None


def test_an_orphaned_gesture_is_closed_by_the_next_activation(monkeypatch):
    """A pane that closes mid-drag never reports the deactivation. The next
    drag anywhere closes the stray, so the deferred eviction cannot stay
    switched off for the rest of the session."""
    first, second = undo.UndoStack(), undo.UndoStack()
    item = _Item(monkeypatch, begin=1, end=99)
    item.frame = 1
    controls.fold_undo(first)
    first.push(_Step())
    first.push(_Step())
    assert first._open_gestures == 1
    controls.fold_undo(second)  # frame 1 again: a fresh activation
    assert first._open_gestures == 0, "the stray was closed"
    assert len(first) == 1, "and folded"
    assert controls._gesture[0] is second


# --- 2. the Sirens doors, drawn -----------------------------------------------


def _scripted_field(
    monkeypatch: pytest.MonkeyPatch, item: _Item, attr: str, label: str, values: list
):
    """The real field, with its answer overridden for one label while the
    scripted drag is on -- there is no pointer in a headless frame.

    ``attr`` is the name on ``controls`` (``"slider_int"``, ``"slider_float"``,
    ``"drag_int"``, ``"color_edit4"``, ``"input_float"``...): every one of
    them is called ``(label, value, ...)`` and answers ``(changed, value)``,
    so one wrapper covers them all.
    """
    real = getattr(controls, attr)
    item.label = label

    def scripted(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        item.last = args[0]
        if args[0] == label and item.dragging():
            return True, values[item.frame - item.begin]
        return result

    monkeypatch.setattr(controls, attr, scripted)


def _scripted_slider(monkeypatch: pytest.MonkeyPatch, item: _Item, label: str, values: list):
    """The real slider, with its answer overridden for one label while the
    scripted drag is on -- there is no pointer in a headless frame.

    ``label`` is the control's **id**, not always the name beside it: a slider
    drawn through ``widgets.labeled_slider_int`` hands this an ``##``-hidden
    id, because imgui draws a slider's label outside the widget to its right
    and a ``-1``-width slider has nowhere to put one. The Sirens callers below
    pass ``"##Tempo"`` and friends for that reason; the picker's channels pass
    their own ids unchanged.
    """
    _scripted_field(monkeypatch, item, "slider_int", label, values)


def _drag(draw_frame: Any, draw: Any, item: _Item) -> None:
    """Frames 1 .. end: the drag, then the release frame."""
    for frame in range(1, item.end + 1):
        item.frame = frame
        draw_frame(draw)


@pytest.mark.parametrize(
    "label,attr,values",
    [("Tempo", "tempo", [121, 124, 130, 133, 140]), ("Speed", "speed", [7, 8, 9, 10, 11])],
)
def test_transport_tempo_and_speed_drags_are_one_step_each(
    monkeypatch, frames, label, attr, values
):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    before = len(tab.doc.history)
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, f"##{label}", values)
    _drag(frames, lambda: sirens_transport.draw(ctx), item)
    assert getattr(tab.doc, attr) == values[-1]
    assert len(tab.doc.history) == before + 1
    assert tab.doc.history.undo(tab.doc)
    assert getattr(tab.doc, attr) != values[-1], "one Ctrl+Z takes the whole drag back"


def test_order_list_rows_drag_is_one_step(monkeypatch, frames):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    pattern = tab.doc.patterns[0]
    sirens_mode.set_caret(ctx, pattern=pattern.uid)
    before = len(tab.doc.history)
    values = [60, 56, 50, 44, 40]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "##Rows", values)
    _drag(frames, lambda: sirens_orders.draw(ctx), item)
    assert pattern.rows == values[-1]
    assert len(tab.doc.history) == before + 1


@pytest.mark.parametrize(
    "label,attr,values",
    [("Tempo", "tempo", [121, 124, 130, 133, 140]), ("Speed", "speed", [7, 8, 9, 10, 11])],
)
def test_effect_tempo_and_speed_drags_are_one_step_each(monkeypatch, frames, label, attr, values):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    state = sirens_mode.ensure(ctx)
    effect = tab.doc.oneshot(state.oneshot)
    before = len(tab.doc.history)
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, f"##{label}", values)
    _drag(frames, lambda: sirens_effects.draw(ctx), item)
    assert getattr(tab.doc.oneshot(effect.uid), attr) == values[-1]
    assert len(tab.doc.history) == before + 1


# --- 3. the Inker picker, drawn -----------------------------------------------


class _PickerDoc:
    def __init__(self) -> None:
        self.palette = [(10, 20, 30, 255), (0, 0, 0, 255)]
        self.is_indexed = True
        self.history = undo.UndoStack()

    def recolour_slot(self, index: int, colour: Any) -> bool:
        self.palette[index] = tuple(colour)
        self.history.push(_Step())
        return True


class _PickerState:
    fg = (10, 20, 30, 255)
    bg = (255, 255, 255, 255)
    fg_slot = 0
    picker_target = "fg"
    palette_usage = None

    def set_fg(self, colour: Any, slot: Any = None) -> None:
        self.fg = tuple(int(c) for c in tuple(colour)[:4])
        self.fg_slot = slot


class _PickerTab:
    def __init__(self) -> None:
        self.doc = _PickerDoc()


def test_a_picker_channel_drag_over_a_palette_slot_is_one_step(monkeypatch, frames):
    tab = _PickerTab()
    state = _PickerState()
    values = [40, 80, 120, 160, 200]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "##Red", values)
    _drag(frames, lambda: inker_picker._rgb(None, state, tab, 0, tab.doc.palette[0]), item)
    assert tab.doc.palette[0][0] == values[-1]
    assert len(tab.doc.history) == 1


def test_a_picker_drag_over_a_free_colour_opens_no_gesture(monkeypatch, frames):
    """Session state, not history: nothing to fold, and nothing left open."""
    tab = _PickerTab()
    state = _PickerState()
    state.fg_slot = None
    values = [40, 80, 120]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "##Red", values)
    _drag(frames, lambda: inker_picker._rgb(None, state, tab, None, state.fg), item)
    assert state.fg[0] == values[-1]
    assert len(tab.doc.history) == 0
    assert tab.doc.history._open_gestures == 0


# --- 4. the popup doors, by source --------------------------------------------


@pytest.mark.parametrize(
    "func,field,write",
    [
        # The three Inker rows' ids gained a "##" prefix in the 2026-09-08
        # label-above pass (the visible label moved to a ``field_label`` line
        # above each field), so the anchor strings below match the id as it
        # reads now rather than the pre-fix source.
        (inker_timeline._group_menu, '"##Opacity##group"', "set_group_props("),
        (inker_timeline._cell_menu, '"##Opacity##cel"', "set_cel_opacity("),
        (inker_timeline._cell_menu, '"##Z##cel"', "set_cel_z("),
        (inker_colors._slots, 'color_edit4("Slot"', "recolour_slot("),
        # The four the 2026-09-04 audit found still unfolded: none of these
        # files called ``fold_undo`` at all, and each setter pushes per report.
        # Ids gained a "##" prefix in the 2026-09-08 consistency pass (the
        # visible label moved to a field_label line above), so the anchor
        # strings below matched the fix, not the pre-fix source.
        (clay_props._material, '"##base colour##bm"', "doc.set_material("),
        (clay_props._material, '"##metallic##bm"', "doc.set_material("),
        (plotter_layers._layer_table, '"##layer-opacity"', "doc.set_layer_props("),
        (plotter_layers._layer_table, '"##layer-tint"', "doc.set_layer_props("),
        (plotter_layers._object_fields, '"##obj-opacity"', "doc.set_object("),
        (sirens_patterns._channel_popup, 'f"Pan##{tag}"', "update_channel("),
        # The 2026-09-05 audit's M06 row: the Terrain tab's colour swatch and
        # its probability field, neither folded, each pushing through
        # ``set_wang_colour`` -> ``_write_wangsets`` per report.
        (plotter_tileset_editor._wang_colours, '"##swatch"', "set_wang_colour("),
        (plotter_tileset_editor._wang_colours, '"Probability"', "set_wang_colour("),
        # The 2026-09-15 audit: three more fields with no fold at all.
        # plotter-03 -- the object layer's Outline colour, unlike every other
        # field in the same table.
        (plotter_layers._layer_table, '"##object-layer-color"', "doc.set_layer_props("),
        # plotter-01 -- a text object's Text/Font/Size/Colour, already one
        # compound ``doc.set_object`` call per changed frame but with no fold,
        # so a typed label still pushed a step per keystroke.
        (plotter_layers._shape_fields, '"##text-value"', "doc.set_object("),
        # plotter-02 -- the Wang set's own name field, unlike the colour name
        # beside it in the same editor.
        (plotter_tileset_editor._wangset_row, '"##tswang-name"', "rename_wangset("),
    ],
    ids=[
        "group-opacity",
        "cel-opacity",
        "cel-z",
        "palette-slot",
        "clay-base-colour",
        "clay-metallic",
        "layer-opacity",
        "layer-tint",
        "object-opacity",
        "channel-pan",
        "wang-colour",
        "wang-probability",
        "object-layer-outline-colour",
        "text-object-fields",
        "wangset-name",
    ],
)
def test_the_popup_doors_fold_between_the_field_and_the_write(func, field, write):
    """These draw only inside ``begin_popup_context_item``, which a headless
    frame cannot open. The invariant is positional: draw, fold, act."""
    source = inspect.getsource(func)
    after_field = source.split(field, 1)[1]
    # ``fold_undo``'s argument differs by pane (``doc.history``,
    # ``tab.doc.history``), so the call itself is what is looked for.
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index(write), f"{write} runs before the fold"


# --- 5. the name fields -------------------------------------------------------


def test_a_typed_name_is_reported_once_when_the_field_is_left(monkeypatch):
    typed = iter(["l", "le", "lea", "lead", "lead"])
    settled = iter([False, False, False, False, True])
    monkeypatch.setattr(widgets.imgui, "input_text", lambda label, value: (True, next(typed)))
    monkeypatch.setattr(widgets.imgui, "is_item_deactivated_after_edit", lambda: next(settled))
    monkeypatch.setattr(widgets, "note_ime_rect", lambda: None)
    seen = [widgets.input_text("Name", "old", commit=True) for _ in range(5)]
    assert seen == ["old", "old", "old", "old", "lead"]


@pytest.mark.parametrize(
    "pane,field_id",
    [
        (sirens_effects, '"##sirens-fx-name"'),
        (sirens_instruments, '"##sirens-inst-name"'),
    ],
    ids=["effects", "instruments"],
)
def test_the_sirens_name_fields_commit_on_release(pane, field_id):
    """The id, not the label, since 2026-09-07: both fields are ``##``-hidden
    with the name drawn above them by ``widgets.field_label``, because a
    ``-1``-width field draws no label at all and "Name" was simply not on
    screen. What is pinned here is unchanged -- a typed name is one rename and
    not six undo steps.

    ``controls.input_text``, not ``widgets.input_text`` (the 2026-09-15
    audit, sirens-04): the latter has no ``enabled``, so both fields moved to
    the one that does, to stop a rename typed mid-save from landing on a
    document a save was already reading. The opener below follows that move
    rather than pinning the pre-fix call.
    """
    source = inspect.getsource(pane)
    opener = "controls.input_text(" + chr(10) + "        " + field_id
    field = source.split(opener, 1)[1].split(")", 1)[0]
    assert "commit=True" in field


@pytest.mark.parametrize(
    "func,field",
    [
        (clay_outliner._row, '"##rename"'),
        (clay_props._identity, '"name##buildname"'),
        (clay_props._palette_row, '"slot name##matname"'),
    ],
    ids=["outliner-rename", "properties-name", "material-slot-name"],
)
def test_renaming_an_object_or_material_slot_in_clay_is_one_undo_step_not_one_per_keystroke(
    func, field
):
    """The 2026-09-06 audit's clay-02: the outliner's rename field, the
    properties panel's name field and the material slot's name field each
    called ``widgets.input_text`` with no ``commit=True``, so
    ``doc.set_props``/``doc.set_material`` -- an unconditional
    ``history.push`` -- fired once per keystroke instead of once per gesture.
    Typing a four-letter name pushed four undo steps and one Ctrl+Z undid one
    typed character rather than the rename."""
    source = inspect.getsource(func)
    after_field = source.split(field, 1)[1]
    call_end = after_field.index(")")
    assert "commit=True" in after_field[:call_end]


# --- 6. the 2026-09-05 audit's own doors ---------------------------------------


def test_an_interrupted_opacity_drag_still_leaves_one_undo_step(monkeypatch, frames):
    """M05: ``inker_menu.header_controls`` used to set ``layer.opacity``
    directly for the live preview and stash the pre-drag value in a bare
    module-level dict keyed by layer uid, pushing the real undo step only on
    ``is_item_deactivated_after_edit``. A control that vanishes first --
    a mode switch, a panel collapse, the document closing -- never renders
    that release frame: the step was never pushed, the dict entry stuck
    around forever, and a later gesture that reused the same uid started
    from that stale value instead of the layer's real one.

    Routed through ``fold_undo`` and a push on every changed frame instead,
    the interruption is survived by the mechanism every other door in this
    file already relies on: whatever gesture is open closes the moment any
    other item activates, wherever that happens to be.
    """
    import numpy as np

    from warlock.kernels import pixel as inker
    from warlock.studio.panes import inker_menu

    doc = inker.Document.from_pixels(np.full((4, 4, 4), 255, dtype=np.uint8))
    before = len(doc.history)
    # Opacity is drawn as a percentage (``labeled_slider_float`` infers
    # ``percent`` for a 0..1 range), so the raw ``controls.slider_float``
    # call this scripts sits in 0..100 space; ``header_controls`` divides
    # back down before it ever reaches the layer.
    values_pct = [80, 60, 40]
    item = _Item(monkeypatch, begin=1, end=1 + len(values_pct))
    _scripted_field(monkeypatch, item, "slider_float", "##Opacity", values_pct)
    # The drag runs, but its own release frame (``item.end``) never comes --
    # the panel is torn down mid-gesture.
    for frame in range(1, item.end):
        item.frame = frame
        frames(lambda: inker_menu.header_controls(None, doc))
    assert doc.stack.active.opacity == pytest.approx(values_pct[-1] / 100.0)
    # Each changed frame already pushed its own step (unlike the old direct
    # assignment, which pushed nothing until release) -- the gesture is still
    # open, so nothing has folded yet, but the edit already exists.
    assert len(doc.history) == before + len(values_pct), "every frame's push landed"
    # Nothing above will ever report the deactivation. What closes a stray
    # gesture is the next activation anywhere -- ``test_an_orphaned_gesture_
    # is_closed_by_the_next_activation`` pins the mechanism itself; this
    # reuses it exactly the way a mode switch followed by any other drag
    # would, on a completely unrelated stack standing in for "the user went
    # on and did something else".
    elsewhere = undo.UndoStack()
    item.frame = item.begin
    controls.fold_undo(elsewhere)
    assert len(doc.history) == before + 1, "the interrupted drag folded to one step"
    assert doc.history.undo(doc)
    assert doc.stack.active.opacity == 1.0, "one Ctrl+Z takes the whole drag back"


def test_the_plotter_layer_list_opacity_row_drag_is_one_step(monkeypatch, frames):
    """M06: ``_opacity_row`` -- the Opacity slider drawn over the layer list
    itself, distinct from ``_layer_table``'s own copy in Properties -- called
    ``doc.set_layer_props`` on every changed frame with no fold at all."""
    from test_plotter_mode import FakeCtx as PlotterFakeCtx
    from test_plotter_mode import _tab as _plotter_tab

    from warlock.studio.panes import plotter_layers

    ctx = PlotterFakeCtx()
    tab = _plotter_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    before = len(tab.doc.history)
    # Percent-scaled, same as the inker opacity door above.
    values_pct = [80, 60, 40, 20]
    item = _Item(monkeypatch, begin=1, end=1 + len(values_pct))
    _scripted_field(monkeypatch, item, "slider_float", "##Opacity", values_pct)
    _drag(frames, lambda: plotter_layers._opacity_row(tab.doc, layer, True), item)
    assert layer.opacity == pytest.approx(values_pct[-1] / 100.0)
    assert len(tab.doc.history) == before + 1
    assert tab.doc.history.undo(tab.doc)


@pytest.mark.parametrize(
    "field,write",
    [
        ('"Columns"', "packwright_mode.set_settings("),
        ('"Padding"', "packwright_mode.set_settings("),
        ('"Extrude"', "packwright_mode.set_settings("),
    ],
    ids=["columns", "padding", "extrude"],
)
def test_packwright_settings_drags_fold_between_the_field_and_the_write(field, write):
    """M06: Columns (a drag_int) and Padding/Extrude (sliders) each called
    ``packwright_mode.set_settings`` on every changed frame, which reaches
    the unconditional history push in ``PackDoc.set_settings``
    (``document.py:410``) once per report -- the audit's reminder that
    Columns is not exempt just because it drags rather than slides.

    By source rather than a drawn frame: all three fields live in the same
    ``draw`` and each folds independently, so a scripted single-item drag
    cannot tell one field's activation from another's the way the real imgui
    item stack does -- ``fold_undo``'s own positional contract (draw, fold,
    act) is what a headless frame cannot exercise here without lying about
    which field the pointer is on, and is exactly what a source check proves
    instead. This is the same reason ``clay_props``'s three material fields
    are pinned this way rather than driven live.
    """
    source = inspect.getsource(packwright_settings.draw)
    after_field = source.split(field, 1)[1]
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index(write), f"{write} runs before the fold"


@pytest.mark.parametrize(
    "field",
    ['"##pivotx"', '"##pivoty"'],
    ids=["pivot-x", "pivot-y"],
)
def test_packwright_sources_pivot_drag_folds_between_the_field_and_the_write(field):
    """packwright-03, the 2026-09-13 audit: ``_pivot_row`` already calls
    ``controls.fold_undo`` correctly (draw, fold, act) for both drag fields,
    but this file's scan -- rows 4 and 6 above -- never included
    ``packwright_sources.py`` at all, so a regression here would have shipped
    silently. This is an *evidence gap*, not a live bug: the fold is already
    in place, so this cannot be made to fail against the current source.
    What it does pin is the same invariant every other row in this file
    pins -- and a scratch copy of ``_pivot_row`` with the fold call deleted
    fails this exact assertion (``ValueError: substring not found``),
    proving the check bites if the fold is ever removed.
    """
    source = inspect.getsource(packwright_sources._pivot_row)
    after_field = source.split(field, 1)[1]
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index("set_pivot("), "set_pivot runs before the fold"


# --- 7. Mason's Properties panel ------------------------------------------------


def test_light_intensity_typed_digit_by_digit_is_one_undo_step(monkeypatch, frames):
    """mason-01, the 2026-09-13 audit: ``_light_block`` called the undoable
    ``doc.set_props`` on every changed frame with no ``controls.fold_undo``,
    so typing "2000" into intensity digit by digit pushed four undo steps and
    one Ctrl+Z left ``200.0`` rather than the pre-edit value.
    """
    from warlock.studio.mason import document as md
    from warlock.studio.mason import nodes as nd

    node = nd.LightNode(uid=nd.new_uid(), name="Lamp", kind="point")
    doc = md.MasonDoc(roots=[node])
    before = len(doc.history)
    values = [2.0, 20.0, 200.0, 2000.0]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_field(monkeypatch, item, "input_float", "##mlightintensity", values)
    _drag(frames, lambda: mason_props._light_block(doc, node), item)
    assert node.intensity == values[-1]
    assert len(doc.history) == before + 1, "one drag folded to one undo step"
    assert doc.history.undo(doc)
    assert node.intensity != values[-1], "one Ctrl+Z takes the whole edit back"


@pytest.mark.parametrize(
    "func,field,write",
    [
        # mason-01: the light, camera and terrain blocks called
        # ``doc.set_props``/``doc.set_terrain_config`` -- both unconditional
        # ``history.push`` -- on every changed frame with no fold at all.
        (mason_props._light_block, '"##mlightintensity"', "doc.set_props("),
        (mason_props._light_block, '"##mlightrange"', "doc.set_props("),
        (mason_props._light_block, '"##mlightinner"', "doc.set_props("),
        (mason_props._light_block, '"##mlightouter"', "doc.set_props("),
        (mason_props._camera_block, '"##mcamfov"', "doc.set_props("),
        (mason_props._camera_block, '"##mcamnear"', "doc.set_props("),
        (mason_props._camera_block, '"##mcamfar"', "doc.set_props("),
        (mason_props._terrain_block, '"##mterrainx"', "doc.set_terrain_config("),
        (mason_props._terrain_block, '"##mterrainz"', "doc.set_terrain_config("),
    ],
    ids=[
        "light-intensity",
        "light-range",
        "light-inner-cone",
        "light-outer-cone",
        "camera-fov",
        "camera-near",
        "camera-far",
        "terrain-size-x",
        "terrain-size-z",
    ],
)
def test_mason_properties_fields_fold_between_the_field_and_the_write(func, field, write):
    """mason-01: the light, camera and terrain numeric fields folded nothing
    -- pinned by source the same way Packwright's Columns/Padding/Extrude are,
    since these are plain ``input_float`` doors rather than drag gestures a
    headless frame can script one at a time."""
    source = inspect.getsource(func)
    after_field = source.split(field, 1)[1]
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index(write), f"{write} runs before the fold"


# --- 8. the shared ``Form`` helper ---------------------------------------------


def test_form_slider_used_for_an_undoable_field_folds_a_multi_frame_drag_into_one_step(
    monkeypatch, frames
):
    """The 2026-09-11 audit's shell-08: ``Form.field`` draws a trailing note
    and an ``imgui.dummy()`` *after* the caller's control, unconditionally, so
    by the time ``Form.slider`` returns to its caller the imgui "last item" is
    that dummy (id 0) rather than the field just drawn. A caller that follows
    every other door's convention and calls ``controls.fold_undo(doc.history)``
    right after ``form_ui.slider(...)`` returns is folding the dummy, which
    never activates or deactivates -- a silent no-op, and every changed frame
    of the drag lands as its own undo step.

    ``Form.slider`` (and ``.number`` and ``.text``) take a ``history=``
    argument instead: the fold happens from inside ``Form.field``'s own
    ``with`` block, immediately after the caller's control and before the
    trailing note/dummy, so the "last item" is still the real field.
    """
    from warlock.studio import forms

    history = undo.UndoStack()
    before = len(history)
    values = [40.0, 80.0, 120.0, 160.0, 200.0]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_field(monkeypatch, item, "slider_float", "##speed", values)

    state = {"value": 10.0}

    def draw() -> None:
        with forms.Form("gesture-test") as form:
            changed, value = form.slider(
                "speed", "Speed", state["value"], 0.0, 200.0, history=history
            )
        if changed:
            state["value"] = value
            history.push(_Step())

    _drag(frames, draw, item)
    assert state["value"] == values[-1]
    assert len(history) == before + 1, "one drag folded to one undo step"
    assert controls._gesture is None
    assert history.undo(None), "one Ctrl+Z takes the whole drag back"
