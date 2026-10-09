"""The palette strip folded into the Inspector's Colour card (Landing 10).

The strip under the viewport (``palette_strip.py``) drew the document's own slots in
one scrolling row; the Colour card now draws them as a wrapped grid under the eight
starter colours, in every state the strip painted in: one shape, several, nothing
selected ("Next shape's colour") and the element modes. Each test's name is the claim;
the ones that fail against the tree as it stood say so:

* the pane module existing, the viewport reserving a strip, ``##clay-palette/add`` and
  ``##palsw*`` labels -- all of them old code;
* the Colour card carrying no ``##slotsw/N`` swatches at all;
* nothing selected showing no card, and a Faces-mode starter repainting the whole shape.
"""

from __future__ import annotations

import importlib
import inspect
import statistics
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from modes.clay.test_inspector import _big_document, _document, _rig
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.studio import anchors, icons, skeletons, widgets
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import starter_colours
from realmspinner.studio.modes.clay.ui import viewport
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import swatches

ADD = f"{icons.PLUS}##slot-add"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    widgets._OPEN_REQUESTS.clear()
    yield
    widgets._OPEN_REQUESTS.clear()


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _with_red(doc: bd.ClayDoc) -> None:
    """Slot 1 is red (the default grey is slot 0)."""
    doc.add_material(replace(bd.default_material("red"), base_color_factor=(1.0, 0.0, 0.0, 1.0)))


def _slot_labels(rig) -> list[str]:
    return [label for label in rig.labels if label.startswith("##slotsw/")]


def _selected(rig) -> list[str]:
    return [c.label for c in rig.driver.controls if c.kind == "swatch" and c.selected]


# -- the strip is gone ---------------------------------------------------------------------------


def test_the_palette_strip_pane_no_longer_exists():
    """Old code: ``panes/palette_strip.py`` was a module, with ``STRIP_H`` and ``draw``."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("realmspinner.studio.modes.clay.ui.panes.palette_strip")


def test_no_clay_column_names_a_palette_pane_and_the_viewport_reserves_no_strip():
    ids = {
        slot.id for column in skeletons.clay(SimpleNamespace()).values() for slot in column.slots
    }
    assert "clay-palette" not in ids and ids == {"clay-inspector"}
    source = inspect.getsource(viewport)
    assert "palette_strip" not in source and "strip_h" not in source, (
        "the centre column's bottom strip is gone: the render owns that height"
    )


# -- the document's slots are in the card ------------------------------------------------------


def test_one_shape_selected_shows_every_document_slot_under_the_starters(imgui_ui):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    rig = _rig(imgui_ui, ctx)
    assert _slot_labels(rig) == ["##slotsw/0", "##slotsw/1"]
    assert ADD in rig.labels
    labels = rig.labels
    assert labels.index("##slotsw/0") > labels.index("##inspsw/Charcoal"), "under the starters"
    assert _selected(rig) == ["##inspsw/custom", "##slotsw/0"], (
        "the shape wears slot 0, a grey that is none of the eight"
    )


def test_several_shapes_selected_show_the_slots_and_outline_a_shared_one(imgui_ui):
    ctx, tab, doc = _document(3)
    _with_red(doc)
    doc.select([o.uid for o in doc.objects])
    rig = _rig(imgui_ui, ctx)
    assert _slot_labels(rig) == ["##slotsw/0", "##slotsw/1"]
    rig.click(label="##slotsw/1")
    assert "##slotsw/1" in _selected(rig), "all three wear slot 1 now, so it is outlined"


def test_nothing_selected_shows_the_card_headed_next_shapes_colour(imgui_ui, monkeypatch):
    """Old code: the strip painted with nothing selected (it set ``last_slot``) but the
    Inspector's nothing-selected state had no Colour card."""
    ctx, tab, doc = _document(1)
    _with_red(doc)
    doc.select([])
    seen: list[str] = []
    real = widgets.section
    monkeypatch.setattr(widgets, "section", lambda label: (seen.append(label), real(label))[1])
    rig = _rig(imgui_ui, ctx)
    assert "Next shape's colour" in seen
    assert _slot_labels(rig) == ["##slotsw/0", "##slotsw/1"]
    assert any(label.startswith("##inspsw/") for label in rig.labels)
    assert not any("##insp-edit-colour" in label for label in rig.labels), (
        "no shape to edit: Edit colour is for a selection"
    )
    assert "clay/palette" in anchors.marked(), "the tour's ring is on the card in this state too"


def test_the_tours_palette_anchor_is_on_the_card_in_every_object_state(imgui_ui):
    ctx, tab, doc = _document(2)
    rig = _rig(imgui_ui, ctx)
    assert "clay/palette" in anchors.marked()
    doc.select([o.uid for o in doc.objects])
    rig.settle()
    assert "clay/palette" in anchors.marked()
    doc.select([])
    rig.settle()
    assert "clay/palette" in anchors.marked()


def test_a_face_mode_card_has_the_starters_and_the_slots_and_vertex_mode_only_the_slots(imgui_ui):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    doc.set_element_mode("face")
    rig = _rig(imgui_ui, ctx)
    assert any(label.startswith("##inspsw/") for label in rig.labels)
    assert _slot_labels(rig) == ["##slotsw/0", "##slotsw/1"]
    doc.set_element_mode("vertex")
    rig.settle()
    assert not any(label.startswith("##inspsw/") for label in rig.labels), (
        "points and edges have nothing to paint, so no starter would do anything"
    )
    assert _slot_labels(rig) == ["##slotsw/0", "##slotsw/1"], "but a slot can still be picked"


# -- clicking ------------------------------------------------------------------------------------


def test_clicking_a_slot_repaints_the_shape_in_one_undo_step_and_remembers_it(imgui_ui):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    obj = doc.objects[0]
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##slotsw/1")
    assert obj.material == 1 and set(obj.mesh.material.tolist()) == {1}
    assert len(doc.history) == steps + 1
    assert tab.last_slot == 1, "the next shape placed by hand wears it"
    assert doc.undo()
    assert obj.material == 0 and set(obj.mesh.material.tolist()) == {0}


def test_a_ctrl_click_on_a_slot_retargets_it_without_painting(imgui_ui, monkeypatch):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    obj = doc.objects[0]
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: True)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##slotsw/1")
    assert tab.edit_slot(doc, obj) == 1, "now the slot Edit colour edits"
    assert obj.material == 0, "not the object's own slot (clay-16)"
    assert set(obj.mesh.material.tolist()) == {0}, "and no face was repainted"
    assert tab.last_slot != 1, "Ctrl+click opens a slot, it does not choose a colour"


def test_clicking_a_slot_with_nothing_selected_only_remembers_it(imgui_ui):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    doc.select([])
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##slotsw/1")
    assert tab.last_slot == 1
    assert len(doc.history) == steps and set(doc.objects[0].mesh.material.tolist()) == {0}
    assert "##slotsw/1" in _selected(rig), "the card outlines the colour the next shape wears"


def test_the_add_slot_button_appends_a_grey_slot_and_paints_the_shape_in_one_step(imgui_ui):
    """Old code: the strip's ``+`` appended a slot and painted nothing."""
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label=ADD)
    assert len(doc.materials) == 2
    assert obj.material == 1 and set(obj.mesh.material.tolist()) == {1}
    assert len(doc.history) == steps + 1, "the slot and the paint are one step"
    assert tab.last_slot == 1
    assert doc.undo()
    assert len(doc.materials) == 1 and obj.material == 0


def test_the_add_slot_button_with_nothing_selected_adds_the_slot_the_next_shape_wears(imgui_ui):
    ctx, tab, doc = _document(1)
    doc.select([])
    rig = _rig(imgui_ui, ctx)
    rig.click(label=ADD)
    assert len(doc.materials) == 2 and tab.last_slot == 1
    assert set(doc.objects[0].mesh.material.tolist()) == {0}, "nothing was painted"


# -- no collisions -------------------------------------------------------------------------------


@pytest.mark.parametrize("selection", ["none", "one", "many"])
def test_no_control_label_in_the_colour_card_is_used_twice(imgui_ui, selection):
    """Tests (and the tour driver) find controls by label; the starter row's
    ``##inspsw/NAME`` and the slot row's ``##slotsw/N`` must never meet, even when a
    starter's click has added a slot of the same name."""
    ctx, tab, doc = _document(3)
    _with_red(doc)
    starter_colours.paint(ctx, doc, [], starter_colours.by_name("Red"))
    picks = {"none": [], "one": [doc.objects[0].uid], "many": [o.uid for o in doc.objects]}
    doc.select(picks[selection])
    rig = _rig(imgui_ui, ctx)
    swatch_labels = [c.label for c in rig.driver.controls if c.kind == "swatch" and c.visible]
    assert len(swatch_labels) == len(set(swatch_labels)), swatch_labels
    starters = {label for label in rig.labels if label.startswith("##inspsw/")}
    assert starters and not set(_slot_labels(rig)) & starters


# -- the element modes -------------------------------------------------------------------------


def test_a_slot_click_in_face_mode_paints_the_picked_faces_only(imgui_ui):
    ctx, tab, doc = _document(1)
    _with_red(doc)
    obj = doc.objects[0]
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[2, 3]))
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##slotsw/1")
    assert list(obj.mesh.material) == [0, 0, 1, 1, 0, 0]
    assert obj.material == 0, "the default slot is not the faces' business"
    assert len(doc.history) == steps + 1


def test_a_starter_click_in_face_mode_paints_the_picked_faces_and_adds_its_slot_in_one_step(
    imgui_ui,
):
    """Old code: the only way to paint faces was the strip; the card's starters
    repainted whole shapes (``repaint_object``) and did not exist in this state."""
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##inspsw/Blue")
    assert [m.name for m in doc.materials][-1] == "Blue"
    assert list(obj.mesh.material) == [1, 0, 0, 0, 0, 0] and obj.material == 0
    assert len(doc.history) == steps + 1, "the new slot and the paint fold together"
    assert doc.undo()
    assert len(doc.materials) == 1 and not obj.mesh.material.any()


def test_a_starter_click_with_no_face_picked_adds_no_slot_and_says_why(imgui_ui):
    ctx, tab, doc = _document(1)
    doc.set_element_mode("face")
    doc.clear_element_sel()
    steps = len(doc.history)
    rig = _rig(imgui_ui, ctx)
    rig.click(label="##inspsw/Blue")
    assert len(doc.materials) == 1 and len(doc.history) == steps, "nothing painted, nothing left"
    assert ctx.toasts, "and the refusal is said"


# -- a long palette ------------------------------------------------------------------------------


def _many_slots(doc: bd.ClayDoc, count: int) -> None:
    while len(doc.materials) < count:
        doc.add_material()


def test_sixty_four_slots_wrap_inside_the_300_px_column_and_never_overflow_sideways(imgui_ui):
    ctx, tab, doc = _document(1)
    _many_slots(doc, 64)
    rig = _rig(imgui_ui, ctx)
    drawn = [c for c in rig.driver.controls if c.label.startswith("##slotsw/") and c.visible]
    assert drawn, "the grid draws"
    assert all(c.rect[0] >= 0.0 and c.rect[0] + c.rect[2] <= 300.0 for c in drawn), (
        "a swatch past the column's right edge is a horizontal overflow"
    )
    rows = {round(c.rect[1]) for c in drawn}
    assert 1 < len(rows) <= swatches.SLOT_ROWS_MAX + 1, "wrapped, and capped at the grid's rows"
    assert len(drawn) < 64, "the rest scroll inside the grid rather than taking the pane"


def test_the_grid_is_as_tall_as_its_rows_cap_and_no_taller_with_sixty_four_slots(imgui_ui):
    ctx, tab, doc = _document(1)
    rig = _rig(imgui_ui, ctx)
    short = rig.find(label="##size##insp").rect[1]
    below_short = rig.find(contains="##insp-edit-colour").rect[1]
    _many_slots(doc, 64)
    rig.settle()
    below_long = rig.find(contains="##insp-edit-colour").rect[1]
    capped = swatches.SLOT_ROWS_MAX * (swatches.SWATCH + 8.0)
    assert below_long - below_short <= capped, "sixty-four slots cost at most the rows cap"
    assert rig.find(label="##size##insp").rect[1] == short, "and nothing above the card moves"


def test_slot_rows_is_pure_and_never_less_than_one_row():
    assert swatches.slot_rows(0, 300.0, 22.0, 8.0) == 1
    assert swatches.slot_rows(9, 276.0, 22.0, 8.0) == 1
    assert swatches.slot_rows(10, 276.0, 22.0, 8.0) == 2
    assert swatches.slot_rows(5, 1.0, 22.0, 8.0) == 5, "one to a row however narrow"


def test_the_card_is_bounded_with_sixty_four_slots_at_any_document_size_under_contention(
    imgui_ui,
):
    ctx, tab, doc = _big_document(4096)
    _many_slots(doc, 64)
    doc.select([doc.objects[0].uid])
    rig = _rig(imgui_ui, ctx, height=1800.0)
    start = time.perf_counter()
    for _ in range(5):
        rig.settle(1)
    assert (time.perf_counter() - start) / 5 < 0.5


@pytest.mark.perf
@pytest.mark.parametrize("shapes", [1, 100, 4096])
def test_the_inspector_frame_stays_cheap_with_sixty_four_slots(imgui_ui, shapes):
    ctx, tab, doc = _big_document(shapes)
    _many_slots(doc, 64)
    budget = {1: 0.005, 100: 0.005, 4096: 0.010}[shapes]
    for selection in ([], [doc.objects[0].uid], [o.uid for o in doc.objects]):
        doc.select(selection)
        rig = _rig(imgui_ui, ctx, height=1800.0)
        samples = []
        for _ in range(12):
            start = time.perf_counter()
            rig.settle(1)
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        where = f"{shapes} shapes, {len(selection)} selected"
        assert median < budget, f"{where}: {median * 1000:.1f} ms"


# -- the pane still fits -------------------------------------------------------------------------


def test_delete_in_the_element_state_clears_the_open_dock_at_1600x950(imgui_ui):
    """Landing 9 left Delete about 8 px short with the dock open at 950 px (the pane
    is about 885 px tall there); the Colour card joined the element state after it, so
    the dock's share and row floor were retuned (0.28 -> 0.25, five rows -> four). Old
    constants fail this."""
    ctx, tab, doc = _document(1)
    doc.set_element_mode("face")
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    widgets.request_open(clay_inspector.DOCK_KEY)
    rig = _rig(imgui_ui, ctx, height=885.0)
    delete = rig.find(contains="##insp-op/delete")
    header = anchors.rect("clay/outliner")
    assert header is not None
    assert delete.rect[1] + delete.rect[3] <= header[1] - 8.0, (
        f"Delete ends at {delete.rect[1] + delete.rect[3]:.0f}, the dock's header starts at "
        f"{header[1]:.0f}"
    )
