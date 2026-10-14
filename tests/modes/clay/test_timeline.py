"""The timeline: its arithmetic, the document doors it writes through, its keys and its strip.

Tranche 3 (U3) of the 2026-10-13 Clay rig plan. The arithmetic (``studio.modes.clay.timeline``)
is asserted on documents; the strip (``ui.panes.timeline``) is drawn in a real imgui context and
pressed. Each claim fails on the tree before the change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pygame
import pytest
from _ui_context import imgui_context
from _ui_driver import UiDriver

from realmspinner.kernels.mesh import anim
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import anchors
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import timeline as tl
from realmspinner.studio.modes.clay.ui.panes import timeline as pane

from .test_clay_mode import FakeCtx

Q0 = np.array([0.0, 0.0, 0.0, 1.0])
Q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _doc(keys=((0.0, 0.0), (1.0, 90.0))) -> tuple[bd.ClayDoc, int, int]:
    """Two bones; the first keyed in rotation at the given (time, degrees) pairs."""
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    arm = _add(doc, "Arm", bone=True, parent=hip, translation=(1, 0, 0))
    doc.set_pose_mode(True)
    for time, degrees in keys:
        half = np.radians(degrees) / 2
        doc.set_rig_time(time)
        doc.key_pose({hip: {"rotation": [0, 0, np.sin(half), np.cos(half)]}})
    doc.set_rig_time(0.0)
    return doc, hip, arm


# -- length, span, frames ------------------------------------------------------------------------


def test_the_length_is_the_last_key_and_the_span_is_whole_seconds_with_room_to_key() -> None:
    doc = bd.ClayDoc()
    assert tl.length(doc) == 0.0 and tl.span(doc) == 1.0
    doc, *_ = _doc(((0.0, 0.0), (1.4, 45.0)))
    assert tl.length(doc) == pytest.approx(1.4) and tl.span(doc) == 2.0
    doc.set_rig_time(2.5)
    assert tl.span(doc) == 3.0, "never short of the playhead"


def test_times_snap_to_whole_frames_at_the_documents_rate() -> None:
    doc, *_ = _doc()
    assert doc.rig_state.fps == 24
    assert tl.frame_of(doc, 0.5) == 12 and tl.time_of(doc, 12) == pytest.approx(0.5)
    assert tl.snap(doc, 0.51) == pytest.approx(0.5)
    doc.set_fps(12)
    assert tl.snap(doc, 0.51) == pytest.approx(6 / 12)
    assert tl.snap(doc, -3.0) == 0.0


# -- rows ----------------------------------------------------------------------------------------


def test_a_bone_has_a_row_when_it_is_keyed_or_selected_and_a_plain_shape_never_does() -> None:
    doc, hip, arm = _doc()
    box = _add(doc, "Box")
    doc.select([box])
    got = tl.rows(doc)
    assert [r.name for r in got] == ["Hip"] and got[0].times == (0.0, 1.0)
    doc.select([arm])
    got = tl.rows(doc)
    assert [(r.name, r.times, r.selected) for r in got] == [
        ("Hip", (0.0, 1.0), False),
        ("Arm", (), True),
    ]


def test_one_diamond_stands_for_all_three_properties_at_a_time() -> None:
    doc, hip, _ = _doc(((0.5, 30.0),))
    doc.set_rig_time(0.5)
    doc.insert_key([hip])
    assert tl.rows(doc)[0].times == (0.5,)


# -- the ruler and the sheet geometry ------------------------------------------------------------


def test_the_ruler_steps_to_a_round_number_that_keeps_labels_apart() -> None:
    ticks = tl.ruler(2.0, 800.0)
    majors = [t for t, _label, major in ticks if major]
    assert majors[:3] == [0.0, 0.25, 0.5], "80 px of label room at 400 px a second"
    assert [label for _t, label, major in ticks if major][:2] == ["0s", "0.25s"]
    wide = tl.ruler(120.0, 400.0)
    majors = [t for t, _l, major in wide if major]
    assert majors[1] > 10.0, "a long clip does not label every second"
    assert tl.ruler(0.0, 100.0) == [] and tl.ruler(1.0, 0.0) == []


def _sheet(rows: int = 3, fit: int = 4, scroll: int = 0) -> tl.Sheet:
    return tl.Sheet(
        x0=100.0, x1=500.0, y0=10.0, ruler_h=20.0, row_h=20.0, visible=2.0,
        gutter=100.0, scroll=scroll, rows_fit=fit, rows=rows,
    )


def test_the_sheet_maps_time_to_pixels_and_back() -> None:
    sheet = _sheet()
    assert sheet.to_x(0.0) == 100.0 and sheet.to_x(2.0) == 500.0 and sheet.to_x(1.0) == 300.0
    assert sheet.to_t(300.0) == pytest.approx(1.0)


def test_the_sheet_finds_the_row_under_a_y_and_nothing_over_the_ruler_or_past_the_last() -> None:
    sheet = _sheet(rows=3, fit=4)
    assert sheet.row_at(15.0) is None, "over the ruler"
    assert sheet.row_at(31.0) == 0 and sheet.row_at(51.0) == 1 and sheet.row_at(70.0) == 2
    assert sheet.row_at(91.0) is None, "past the last of three rows"
    scrolled = _sheet(rows=10, fit=4, scroll=3)
    assert scrolled.row_at(31.0) == 3


def test_the_scroll_is_clamped_to_what_exists() -> None:
    assert _sheet(rows=3, fit=4, scroll=5).scroll == 0
    assert _sheet(rows=10, fit=4, scroll=99).scroll == 6
    assert _sheet(rows=10, fit=4, scroll=-2).scroll == 0


def test_a_press_near_a_diamond_finds_it_and_a_press_elsewhere_does_not() -> None:
    doc, hip, _ = _doc()
    sheet = _sheet(rows=1)
    rows = tl.rows(doc)
    assert tl.nearest_key(rows, sheet.to_x(1.0) + 3.0, 0, sheet.to_x) == (hip, 1.0)
    assert tl.nearest_key(rows, sheet.to_x(0.5), 0, sheet.to_x) is None
    assert tl.nearest_key(rows, 0.0, 5, sheet.to_x) is None


# -- the transport -------------------------------------------------------------------------------


def test_play_needs_something_to_play_and_a_finished_one_shot_starts_again() -> None:
    bare = bd.ClayDoc()
    assert not tl.toggle_play(bare) and not bare.playing
    doc, *_ = _doc()
    doc.set_animation_loop(0, False)
    assert tl.toggle_play(doc) and doc.playing
    assert not tl.toggle_play(doc) and not doc.playing
    doc.set_rig_time(1.0)
    assert tl.toggle_play(doc)
    assert doc.rig_time == 0.0, "pressing Play on the finished one-shot goes back to the start"


def test_playback_advances_by_the_frame_time_and_wraps_a_loop() -> None:
    doc, *_ = _doc()
    tl.toggle_play(doc)
    assert tl.tick(doc, 0.25) and doc.rig_time == pytest.approx(0.25)
    assert tl.tick(doc, 0.5) and doc.rig_time == pytest.approx(0.75)
    assert tl.tick(doc, 0.5) and doc.rig_time == pytest.approx(0.25), "a loop wraps past its end"
    assert doc.playing


def test_a_one_shot_stops_on_its_last_key() -> None:
    doc, *_ = _doc()
    doc.set_animation_loop(0, False)
    tl.toggle_play(doc)
    tl.tick(doc, 3.0)
    assert doc.rig_time == pytest.approx(1.0) and not doc.playing


def test_playback_stops_by_itself_when_the_animation_goes_or_pose_mode_is_left() -> None:
    doc, *_ = _doc()
    tl.toggle_play(doc)
    doc.delete_animation(0)
    assert not tl.tick(doc, 0.1) and not doc.playing
    doc, *_ = _doc()
    tl.toggle_play(doc)
    doc.set_pose_mode(False)
    assert not doc.playing, "leaving Pose mode stops it"
    doc.playing = True
    assert not tl.tick(doc, 0.1) and not doc.playing


def test_playing_is_never_an_undo_step_and_drops_an_unkeyed_pose() -> None:
    doc, hip, _ = _doc()
    steps = len(doc.history)
    doc.set_live_pose(hip, "translation", [0, 5, 0])
    tl.toggle_play(doc)
    tl.tick(doc, 0.1)
    assert len(doc.history) == steps and doc.live_pose == {}


def test_stepping_and_jumping_move_the_playhead_on_frames_and_stop_playback() -> None:
    doc, *_ = _doc()
    tl.toggle_play(doc)
    tl.step(doc, 1)
    assert not doc.playing and doc.rig_time == pytest.approx(1 / 24)
    tl.step(doc, -5)
    assert doc.rig_time == 0.0, "never before the start"
    tl.go_end(doc)
    assert doc.rig_time == pytest.approx(1.0)
    tl.go_start(doc)
    assert doc.rig_time == 0.0


def test_the_playhead_at_a_looping_clips_length_shows_its_last_key_not_its_first() -> None:
    doc, hip, _ = _doc()
    assert doc.animation().loop
    doc.set_rig_time(1.0)
    assert np.allclose(doc.pose_overrides()[hip]["rotation"], Q90, atol=1e-6)
    doc.set_rig_time(1.25)
    wrapped = anim.pose(doc.animation(), 0.25)[hip]["rotation"]
    assert np.allclose(doc.pose_overrides()[hip]["rotation"], wrapped)


# -- moving and deleting diamonds ----------------------------------------------------------------


def test_sliding_a_diamond_moves_all_three_properties_as_one_step() -> None:
    doc, hip, _ = _doc(((0.0, 0.0), (1.0, 90.0)))
    doc.set_rig_time(1.0)
    doc.insert_key([hip])
    steps = len(doc.history)
    assert doc.move_bone_keys([(hip, 1.0)], 0.5)
    assert len(doc.history) == steps + 1
    channels = doc.animation().channels[hip]
    assert {p: list(c.times) for p, c in channels.items()} == {
        "rotation": [0.0, 1.5],
        "translation": [1.5],
        "scale": [1.5],
    }


def test_a_slide_that_would_leave_the_clip_stops_at_zero_and_one_onto_another_replaces_it() -> None:
    doc, hip, _ = _doc(((0.5, 0.0), (1.0, 90.0)))
    assert doc.move_bone_keys([(hip, 0.5)], -9.0)
    assert list(doc.animation().channels[hip]["rotation"].times) == [0.0, 1.0]
    assert doc.move_bone_keys([(hip, 0.0)], 1.0)
    assert list(doc.animation().channels[hip]["rotation"].times) == [1.0]


def test_a_slide_of_nothing_or_with_no_animation_pushes_nothing() -> None:
    doc, hip, _ = _doc()
    steps = len(doc.history)
    assert not doc.move_bone_keys([(hip, 1.0)], 0.0)
    assert not doc.move_bone_keys([], 1.0)
    assert len(doc.history) == steps
    assert not bd.ClayDoc().move_bone_keys([(1, 0.0)], 1.0)


def test_deleting_diamonds_removes_every_property_there_as_one_step() -> None:
    doc, hip, _ = _doc()
    doc.set_rig_time(1.0)
    doc.insert_key([hip])
    steps = len(doc.history)
    assert doc.delete_bone_keys([(hip, 1.0)])
    assert len(doc.history) == steps + 1
    assert list(doc.animation().channels[hip]["rotation"].times) == [0.0]
    assert "translation" not in doc.animation().channels[hip]
    assert not doc.delete_bone_keys([(hip, 1.0)]), "already gone"


# -- keys ----------------------------------------------------------------------------------------


def _key(ctx: FakeCtx, key: int) -> bool:
    return clay_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))


def _tab_with_rig(svc) -> tuple[FakeCtx, object, bd.ClayDoc, int]:
    ctx = FakeCtx(svc)
    doc, hip, _arm = _doc()
    tab = clay_mode.adopt(ctx, doc, title="Rig")
    return ctx, tab, doc, hip


def test_space_plays_home_and_end_jump_and_the_arrows_step_in_pose_mode(svc) -> None:
    ctx, tab, doc, _hip = _tab_with_rig(svc)
    assert _key(ctx, pygame.K_SPACE) and doc.playing
    assert _key(ctx, pygame.K_SPACE) and not doc.playing
    assert _key(ctx, pygame.K_END) and doc.rig_time == pytest.approx(1.0)
    assert _key(ctx, pygame.K_HOME) and doc.rig_time == 0.0
    assert _key(ctx, pygame.K_RIGHT) and doc.rig_time == pytest.approx(1 / 24)
    assert _key(ctx, pygame.K_LEFT) and doc.rig_time == 0.0


def test_those_keys_do_nothing_to_the_playhead_outside_pose_mode(svc) -> None:
    ctx, tab, doc, _hip = _tab_with_rig(svc)
    doc.set_pose_mode(False)
    _key(ctx, pygame.K_SPACE)
    _key(ctx, pygame.K_RIGHT)
    assert not doc.playing and doc.rig_time == 0.0


def test_delete_removes_the_picked_diamonds_and_x_does_too(svc) -> None:
    ctx, tab, doc, hip = _tab_with_rig(svc)
    tab.key_sel = [(hip, 1.0)]
    steps = len(doc.history)
    assert _key(ctx, pygame.K_DELETE)
    assert tab.key_sel == [] and len(doc.history) == steps + 1
    assert list(doc.animation().channels[hip]["rotation"].times) == [0.0]
    tab.key_sel = [(hip, 0.0)]
    _key(ctx, pygame.K_x)
    assert doc.animation().channels == {}


def test_delete_with_no_diamond_picked_is_the_ordinary_delete(svc) -> None:
    ctx, tab, doc, hip = _tab_with_rig(svc)
    doc.select([hip])
    _key(ctx, pygame.K_DELETE)
    assert not any(o.uid == hip for o in doc.objects), "the bone was deleted as an object"


# -- the strip, drawn ----------------------------------------------------------------------------


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Prompts:
    def __init__(self) -> None:
        self.asked: list[object] = []

    def ask(self, prompt: object) -> None:
        self.asked.append(prompt)


class Strip:
    """The timeline strip drawn 900 px wide, with a driver that presses it."""

    def __init__(self, imgui, doc: bd.ClayDoc, *, saving: bool = False) -> None:
        self.imgui = imgui
        self.doc = doc
        self.ctx = SimpleNamespace(prompts=_Prompts(), toasts=[])
        self.ctx.toast = lambda message, level="info", *a: self.ctx.toasts.append((message, level))
        self.tab = SimpleNamespace(
            doc=doc, saving=saving, key_sel=[], key_drag=None, timeline_scroll=0
        )
        imgui.get_io().display_size = (1600.0, 950.0)
        self.driver = UiDriver(imgui, self._frame)
        self.driver.settle(3)

    def _frame(self) -> None:
        imgui = self.imgui
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        anchors.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((1000, 400))
        imgui.begin("##timeline-host", None, flags)
        pane.draw(self.ctx, self.tab, 900.0, pane.height(self.doc))
        imgui.end()
        imgui.end_frame()

    def click(self, **kw) -> None:
        self.driver.click(self.driver.find(**kw))
        self.driver.settle(2)

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.driver.controls if c.visible]


def test_the_strip_takes_height_only_in_pose_mode() -> None:
    doc = bd.ClayDoc()
    assert pane.height(doc) == 0.0
    doc.set_pose_mode(True)
    assert pane.height(doc) > 0.0


def test_the_strip_draws_the_transport_the_list_and_the_settings(imgui_ui) -> None:
    doc, *_ = _doc()
    strip = Strip(imgui_ui, doc)
    text = " ".join(strip.labels)
    for needle in ("|<", "Play", ">|", "New", "clay-timeline-anim", "clay-timeline-fps"):
        assert needle in text, needle
    assert "Loop##clay-timeline-loop" in strip.labels
    assert "Auto-key##clay-timeline-autokey" in strip.labels


def test_play_in_the_strip_starts_playback_and_is_greyed_with_nothing_to_play(imgui_ui) -> None:
    doc, *_ = _doc()
    strip = Strip(imgui_ui, doc)
    strip.click(contains="Play##clay-timeline-bar")
    assert doc.playing

    bare = bd.ClayDoc()
    bare.set_pose_mode(True)
    other = Strip(imgui_ui, bare)
    assert not other.driver.find(contains="Play##clay-timeline-bar").enabled


def test_new_copy_and_delete_in_the_strip_edit_the_animation_list(imgui_ui) -> None:
    doc, *_ = _doc()
    strip = Strip(imgui_ui, doc)
    strip.click(contains="New##clay-timeline-bar")
    assert [a.name for a in doc.rig_state.animations] == ["Animation", "Animation.001"]
    assert doc.active_animation == 1
    strip.click(contains="Copy##clay-timeline-bar")
    assert len(doc.rig_state.animations) == 3
    strip.click(contains="Delete##clay-timeline-bar")
    assert len(doc.rig_state.animations) == 2


def test_rename_asks_for_a_name_and_applies_it(imgui_ui) -> None:
    doc, *_ = _doc()
    strip = Strip(imgui_ui, doc)
    strip.click(contains="Rename##clay-timeline-bar")
    [prompt] = strip.ctx.prompts.asked
    assert prompt.value == "Animation"
    prompt.on_accept("Walk")
    assert doc.animation().name == "Walk"
    prompt.on_accept("Walk")  # the same name is no change, not a refusal
    assert strip.ctx.toasts == []


def test_a_refused_edit_in_the_strip_is_a_toast_not_a_crash(imgui_ui) -> None:
    doc, *_ = _doc()
    doc.add_animation("Run")
    strip = Strip(imgui_ui, doc)
    strip.click(contains="Rename##clay-timeline-bar")
    [prompt] = strip.ctx.prompts.asked
    prompt.on_accept("Animation")  # taken by the first
    assert any("already" in message for message, _level in strip.ctx.toasts)


def test_the_settings_write_through_the_documents_doors(imgui_ui) -> None:
    doc, *_ = _doc()
    strip = Strip(imgui_ui, doc)
    strip.click(contains="Loop##clay-timeline-loop")
    assert doc.animation().loop is False
    strip.click(contains="Auto-key##clay-timeline-autokey")
    assert doc.auto_key is False


def test_the_strip_reads_the_playhead_in_seconds_and_frames(imgui_ui) -> None:
    doc, *_ = _doc()
    doc.set_rig_time(0.5)
    assert pane._readout(doc) == "0.50 s  ·  frame 12"


def test_pressing_the_ruler_scrubs_and_a_diamond_picks_and_starts_a_slide() -> None:
    doc, hip, _ = _doc()
    tab = SimpleNamespace(key_sel=[], key_drag=None)
    rows = tl.rows(doc)
    sheet = _sheet(rows=len(rows), fit=4)
    sheet.y0, sheet.ruler_h = 10.0, 20.0

    started = pane._begin(tab, doc, rows, sheet, (sheet.to_x(0.5), 15.0), False)
    assert started == {"mode": "scrub"} and doc.rig_time == pytest.approx(0.5)

    started = pane._begin(tab, doc, rows, sheet, (sheet.to_x(1.0) + 2.0, 40.0), False)
    assert started["mode"] == "keys" and tab.key_sel == [(hip, 1.0)]
    assert doc.rig_time == pytest.approx(1.0), "picking a diamond parks the playhead on it"

    pane._begin(tab, doc, rows, sheet, (sheet.to_x(0.0), 40.0), True)
    assert set(tab.key_sel) == {(hip, 0.0), (hip, 1.0)}, "Shift adds"
    pane._begin(tab, doc, rows, sheet, (sheet.to_x(0.0), 40.0), True)
    assert tab.key_sel == [(hip, 1.0)], "Shift on a picked one takes it away"

    pane._begin(tab, doc, rows, sheet, (sheet.to_x(0.5), 40.0), False)
    assert tab.key_sel == [], "empty track clears the pick"
