"""The Clay torch tour: it exists, it gates, and each part check means what it says.

Each test's name is the claim. The conditions are read against a *real*
``ClayDoc`` built through Clay's own add door (``tools.add_primitive``) rather
than a stub, because a check that reads the wrong attribute passes a stub and
fails a person.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import menus
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools
from realmspinner.studio.panes import model_gate
from realmspinner.studio.panes import tour as tour_pane
from realmspinner.studio.state import TourState
from realmspinner.studio.tour import TOURS, find

KEY = "clay-torch"


class _Settings:
    def __init__(self) -> None:
        self.saved: dict[str, object] = {}

    def get(self, key, default=None):
        return self.saved.get(key, default)

    def set(self, key, value):
        self.saved[key] = value


def _ctx():
    return SimpleNamespace(
        state=SimpleNamespace(mode="home", clay=None, inker=None, sirens=None, tour=TourState()),
        settings=_Settings(),
        toasts=[],
        toast=lambda *a, **k: None,
        cache=SimpleNamespace(invalidate=lambda: None),
    )


def _sat(ctx, name, arg=None):
    return tour_pane.satisfied(ctx, name, arg)


def _tab(ctx):
    return ctx.state.clay.active


def test_the_clay_torch_tour_is_in_the_registry_and_belongs_to_clay():
    tour = find(KEY)
    assert tour is not None
    assert tour in TOURS
    assert tour.mode == "clay"
    assert TOURS[-1] is tour, "offered last, so the older tours keep their order"


def test_the_clay_torch_tour_reaches_the_help_menu_through_the_registry():
    assert menus._COMMAND_PATHS[f"tour:{KEY}"] == ("Help",)


def test_only_the_clay_torch_tour_gates_next():
    gated = {tour.key for tour in TOURS for step in tour.steps if step.gate}
    assert gated == {KEY}


def test_a_gated_step_waits_on_a_real_condition_and_an_ungated_one_always_advances():
    for tour in TOURS:
        for step in tour.steps:
            if step.gate:
                assert step.done.name != "manual", f"{tour.key}/{step.id} gates on nothing"
            assert tour_pane.can_advance(step, False) is (not step.gate)
            assert tour_pane.can_advance(step, True) is True


def test_every_clay_torch_building_step_is_gated():
    tour = find(KEY)
    for step in tour.steps:
        assert step.gate == (step.done.name != "manual"), step.id


def test_no_clay_condition_holds_before_clay_has_a_document():
    ctx = _ctx()
    tour = find(KEY)
    for step in tour.steps:
        if step.done.name.startswith("clay_"):
            assert not _sat(ctx, step.done.name, step.done.arg), step.id


def test_an_empty_model_check_is_false_until_a_document_exists_and_false_once_it_has_a_shape():
    ctx = _ctx()
    assert not _sat(ctx, "clay_empty_model")
    clay_mode.new_document(ctx)
    assert _sat(ctx, "clay_empty_model")
    clay_tools.add_primitive(ctx, _tab(ctx).doc, "box")
    assert not _sat(ctx, "clay_empty_model")


def test_the_box_check_stays_incomplete_until_a_box_exists():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    assert not _sat(ctx, "clay_shapes_at_least", "box:1")
    clay_tools.add_primitive(ctx, _tab(ctx).doc, "cylinder")
    assert not _sat(ctx, "clay_shapes_at_least", "box:1")
    clay_tools.add_primitive(ctx, _tab(ctx).doc, "box")
    assert _sat(ctx, "clay_shapes_at_least", "box:1")


def test_two_cylinders_are_needed_for_the_collar_and_one_is_not_enough():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    doc = _tab(ctx).doc
    clay_tools.add_primitive(ctx, doc, "cylinder")
    assert not _sat(ctx, "clay_shapes_at_least", "cylinder:2")
    clay_tools.add_primitive(ctx, doc, "cylinder")
    assert _sat(ctx, "clay_shapes_at_least", "cylinder:2")


def test_a_freshly_added_shape_is_not_shaped_until_it_is_resized_or_reparameterised():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    doc = _tab(ctx).doc
    box = clay_tools.add_primitive(ctx, doc, "box")
    cone = clay_tools.add_primitive(ctx, doc, "cone")
    assert not _sat(ctx, "clay_shape_placed", "box")
    assert not _sat(ctx, "clay_shape_placed", "cone")

    box.scale[:] = (0.2, 0.4, 0.04)  # resized
    assert _sat(ctx, "clay_shape_placed", "box")
    assert not _sat(ctx, "clay_shape_placed", "cone")

    cone.params["segments"] = 5  # reparameterised
    assert _sat(ctx, "clay_shape_placed", "cone")

    box.scale[:] = 1.0
    # Moved is not shaped: the Add door stands a new shape on the ground beside
    # what is there, so a non-zero translation is what an untouched shape has.
    box.translation[:] = (0.0, 0.5, 0.0)
    assert not _sat(ctx, "clay_shape_placed", "box")


def test_a_shape_the_click_door_grounded_is_still_not_shaped_until_the_reader_shapes_it():
    """The regression the redefinition exists for: ``place_primitive`` puts a box
    at ``y = 0.5``, which the old test (any non-origin translation is shaped)
    read as the reader having already shaped it, so the torch tour's size step
    was done before they typed a number."""
    ctx = _ctx()
    clay_mode.new_document(ctx)
    doc = _tab(ctx).doc
    placed = clay_tools.place_primitive(ctx, doc, "box")
    assert placed.translation[1] != 0.0, "the click door no longer grounds the shape"
    assert not _sat(ctx, "clay_shape_placed", "box")
    placed.scale[:] = (0.2, 0.4, 0.04)
    assert _sat(ctx, "clay_shape_placed", "box")


def test_a_shape_check_names_a_kind_the_registry_knows_or_is_false():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    assert not _sat(ctx, "clay_shape_placed", "no-such-shape")
    assert not _sat(ctx, "clay_shapes_at_least", "")
    assert not _sat(ctx, "clay_shapes_at_least", "box:many")


def test_the_palette_check_needs_slots_beyond_the_one_a_new_model_starts_with():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    doc = _tab(ctx).doc
    assert not _sat(ctx, "clay_palette_at_least", "3")
    doc.add_material()
    doc.add_material()
    assert _sat(ctx, "clay_palette_at_least", "3")
    assert not _sat(ctx, "clay_palette_at_least", "three")


def test_painting_with_the_first_slot_does_not_count_and_another_slot_does():
    ctx = _ctx()
    clay_mode.new_document(ctx)
    doc = _tab(ctx).doc
    a = clay_tools.add_primitive(ctx, doc, "cylinder")
    b = clay_tools.add_primitive(ctx, doc, "cone")
    doc.add_material()
    doc.repaint_object(a.uid, 0)
    doc.repaint_object(b.uid, 0)
    assert not _sat(ctx, "clay_painted_at_least", "2")
    doc.repaint_object(a.uid, 1)
    assert not _sat(ctx, "clay_painted_at_least", "2")
    doc.repaint_object(b.uid, 1)
    assert _sat(ctx, "clay_painted_at_least", "2")


def test_a_new_untitled_model_is_not_saved_even_though_it_has_no_unsaved_changes():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    assert not tab.dirty
    assert not _sat(ctx, "clay_saved")


def test_a_saved_model_stops_being_saved_the_moment_it_is_edited_again():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    clay_tools.add_primitive(ctx, tab.doc, "box")
    tab.path = Path("torch.rblk")
    tab.mark_saved()
    assert _sat(ctx, "clay_saved")
    clay_tools.add_primitive(ctx, tab.doc, "cone")
    assert not _sat(ctx, "clay_saved")


def test_only_an_export_to_the_library_completes_the_export_step():
    ctx = _ctx()
    tab = clay_mode.new_document(ctx)
    assert not _sat(ctx, "clay_exported")
    tab.job_id = "job-1"
    assert _sat(ctx, "clay_exported")


def test_every_shape_the_tour_asks_for_is_one_clay_actually_offers():
    tour = find(KEY)
    for step in tour.steps:
        if step.done.name == "clay_shape_placed":
            assert step.done.arg in bp.CLAY_GENERATOR_NAMES
        if step.done.name == "clay_shapes_at_least":
            assert step.done.arg.partition(":")[0] in bp.CLAY_GENERATOR_NAMES


# -- walking the whole tour ---------------------------------------------------


def _milestones(ctx):
    """What a reader does on each step of the tour, by step id. Steps that only
    explain are absent."""

    def box_scale():
        _obj("box").scale[:] = (0.2, 0.4, 0.04)

    def _obj(kind):
        return next(o for o in _tab(ctx).doc.objects if o.generator == kind)

    def add(kind):
        return lambda: clay_tools.add_primitive(ctx, _tab(ctx).doc, kind)

    def shaft():
        _obj("cylinder").scale[:] = (0.08, 0.6, 0.08)

    def flame():
        _obj("cone").params["segments"] = 5

    def colours():
        _tab(ctx).doc.add_material()
        _tab(ctx).doc.add_material()

    def paint():
        doc = _tab(ctx).doc
        doc.repaint_object(_obj("cylinder").uid, 1)
        doc.repaint_object(_obj("cone").uid, 2)

    def save():
        tab = _tab(ctx)
        tab.path = Path("torch.rblk")
        tab.mark_saved()

    def export():
        _tab(ctx).job_id = "job-1"

    return {
        "open-clay": lambda: setattr(ctx.state, "mode", "clay"),
        "new-model": lambda: clay_mode.new_document(ctx),
        "plate-add": add("box"),
        "plate-shape": box_scale,
        "shaft": lambda: (add("cylinder")(), shaft()),
        "collar": add("cylinder"),
        "flame-add": add("cone"),
        "flame-shape": flame,
        "palette-colours": colours,
        "paint": paint,
        "save": save,
        "export": export,
    }


def test_the_whole_torch_tour_can_be_walked_to_the_end_and_is_remembered(monkeypatch):
    monkeypatch.setattr(model_gate, "mode_gate", lambda ctx, mode: ("", ()))
    ctx = _ctx()
    tour = find(KEY)
    ctx.state.tour.start(KEY)
    milestones = _milestones(ctx)

    for index, step in enumerate(tour.steps):
        assert ctx.state.tour.index == index
        ctx.state.mode = step.mode or ctx.state.mode
        if step.gate:
            ctx.state.tour.satisfied = _sat(ctx, step.done.name, step.done.arg)
            assert not tour_pane.can_advance(step, ctx.state.tour.satisfied), (
                f"{step.id}: Next is open before the reader has done anything"
            )
            milestones[step.id]()
        ctx.state.tour.satisfied = _sat(ctx, step.done.name, step.done.arg)
        assert tour_pane.can_advance(step, ctx.state.tour.satisfied), (
            f"{step.id}: the milestone was applied and Next is still shut"
        )
        tour_pane.advance(ctx)

    assert not ctx.state.tour.running
    assert KEY in ctx.state.tour.finished
    assert ctx.settings.saved["tours_finished"] == [KEY]


# -- skipping and closing leave the document alone -----------------------------


def _snapshot(ctx):
    tab = _tab(ctx)
    return (
        tab.doc.history.head,
        [(o.uid, o.generator, tuple(o.scale), tuple(o.translation)) for o in tab.doc.objects],
        len(tab.doc.materials),
        tab.path,
        tab.job_id,
    )


@pytest.mark.parametrize("how", ["skip", "back", "close"])
def test_skipping_going_back_or_closing_the_tour_does_not_touch_the_document(monkeypatch, how):
    monkeypatch.setattr(model_gate, "mode_gate", lambda ctx, mode: ("", ()))
    ctx = _ctx()
    ctx.state.mode = "clay"
    tab = clay_mode.new_document(ctx)
    clay_tools.add_primitive(ctx, tab.doc, "box")
    ctx.state.tour.start(KEY)
    tour_pane.advance(ctx)
    tour_pane.advance(ctx)
    before = _snapshot(ctx)

    if how == "skip":
        # A gated card the reader has not satisfied: Skip step is advance().
        step = find(KEY).step(ctx.state.tour.index)
        assert not tour_pane.can_advance(step, False) or not step.gate
        tour_pane.advance(ctx)
    elif how == "back":
        tour_pane.advance(ctx, -1)
    else:
        tour_pane.stop(ctx)

    assert _snapshot(ctx) == before
    assert (how == "close") == (not ctx.state.tour.running)
