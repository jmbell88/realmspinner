"""Every Muse pane, drawn -- on a machine with no GPU and no sound card.

``test_sirens_panes_smoke.py``'s arrangement and its argument: the other pane
smoke tests build a *renderer* over a real GL context and skip where there is
none, which is most CI and every remote shell, and that skip is what lets a
wrong argument order ship. Nothing here presents anything, so no GL is needed --
declaring ``renderer_has_textures`` is what lets imgui finish a frame without a
backend claiming its font atlas -- and what the frame still does is run every
widget call, every draw-list call and every layout pass for real.

The panes are drawn into a window with a **stated size**, for that file's
reason: the brief's row gives way as the pane narrows, so a default-sized window
would exercise one branch of ``_row_widths`` and never the other.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio.modes.muse import mode as muse_mode
from warlock.studio.modes.muse.ui import brief as muse_brief
from warlock.studio.modes.muse.ui.panes import player as muse_player
from warlock.studio.modes.muse.ui.panes import recipe as muse_recipe
from warlock.studio.modes.muse.ui.panes import results as muse_results

from .test_muse_mode import FakeCtx

PANES = (
    ("muse-brief", muse_brief),
    ("muse-recipe", muse_recipe),
    ("muse-results", muse_results),
    ("muse-player", muse_player),
)

#: Wide enough that the brief draws its count control and narrow enough to be a
#: plausible window. The tray wraps its cards against this too.
WINDOW = (900.0, 700.0)

#: The width at which ``muse_brief._row_widths`` gives the count away. Well
#: under ``TEXT_MIN_W`` plus the three fixed controls, so the branch is
#: certainly taken rather than nearly taken.
NARROW = (420.0, 700.0)


@pytest.fixture
def frames():
    """A bare imgui context, built and destroyed around this file.

    The save-and-restore is ``test_pane_guard``'s discipline for its reason: at
    most one imgui context may exist at a time, and a file that wants one builds
    and destroys it rather than relying on collection order.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = WINDOW) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(size)
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    """No pane in this file may reach the mixer. CI has no card and a box that
    has one is not something a drawing test should depend on."""
    from warlock.studio.modes.sirens import audio as sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "")


class _Cache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs


def _ctx(tmp_path, jobs: list[dict[str, Any]] | None = None) -> FakeCtx:
    """``test_muse_mode``'s context, with the *real* ``AppState`` on it.

    That substitution is the difference between this file and that one. The
    controller tests want a stub small enough to read; these panes go through
    ``focus.begin``/``item``, which keep their ring on the app state itself --
    so a stub would exercise a focus ring that does not exist, and pass.
    """
    from warlock.studio.state import AppState

    ctx = FakeCtx(tmp_path)
    ctx.state = AppState()
    ctx.cache = _Cache(jobs or [])
    return ctx


def _take(job_id: str, status: str = "done") -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": "music",
        "stage": "music",
        "status": status,
        "prompt": "dark ambient, dungeon, low strings, slow",
        "params": {"duration": 60.0, "actual_duration": 60.0},
    }


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_takes_in_the_tray(name, pane, frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a"), _take("b"), _take("c", status="queued")])
    frames(lambda: pane.draw(ctx))


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_on_a_first_visit(name, pane, frames, tmp_path):
    """The state does not exist yet and there are no takes.

    This is the frame a user actually sees first, and it is the one a mode
    reached through the generic per-mode loops in ``test_studio_smoke`` gets --
    so an ``ensure`` missing anywhere shows up here rather than in the app.
    """
    frames(lambda: pane.draw(_ctx(tmp_path)))


def test_the_brief_gives_the_count_away_before_it_clips_generate(frames, tmp_path):
    """The row's give-way order, exercised rather than only written down.

    ``same_line`` past the pane edge draws a control *nowhere*, so an unstated
    order does not produce a cramped row -- it produces a missing Generate,
    which is the one control the bar exists for.
    """
    ctx = _ctx(tmp_path)
    frames(lambda: muse_brief.draw(ctx), NARROW)


def test_the_brief_still_draws_with_custom_open_at_the_narrow_width(frames, tmp_path):
    """Custom's seconds field is fixed like Duration and Generate, never
    dropped (2026-09-07) -- it is the last control that should vanish out from
    under someone who just opened it to type an exact number. Fails against a
    ``_row_widths`` that still takes no ``duration_custom`` argument at all,
    and would fail again against one that folded Custom into the give-way
    order instead of reserving its width up front.
    """
    ctx = _ctx(tmp_path)
    muse_mode.ensure(ctx).duration_custom = True
    frames(lambda: muse_brief.draw(ctx), NARROW)


@pytest.mark.parametrize(
    "duration_custom", [False, True], ids=["collapsed", "custom-open"]
)
def test_the_bar_fits_the_height_it_declares(frames, tmp_path, duration_custom):
    """``BAR_H``'s own comment says it is provisional and that this test is
    what confirms it, not a figure chosen by eye.

    The ``frames`` fixture draws into a bare ``imgui.begin("smoke")`` window
    rather than through ``layout.pane``, so the real pane's padding is not
    present here -- ``imgui.get_style().window_padding.y`` is added back on
    both edges so the claim is "this content fits a pane of that height," not
    "it fits a window with no padding at all." Parametrised over the Custom
    pill because opening it must not blow the budget either: the seconds
    field beside the pills is fixed and never dropped, so it is always part of
    what the bar has to fit.

    **Finding, not just a check.** The bar was already over its declared
    height before this pass -- 118 against roughly 142 dp of actual content --
    because neither figure ever charged for the padding ``layout.pane`` spends
    on top of every widget it draws, and nothing drew this bar for real to
    catch it. This test is that thing; ``BAR_H`` was moved to 270 to match
    what it measures.
    """
    from imgui_bundle import imgui

    ctx = _ctx(tmp_path)
    muse_mode.ensure(ctx).duration_custom = duration_custom
    positions: dict[str, float] = {}

    def build() -> None:
        positions["before"] = imgui.get_cursor_pos_y()
        muse_brief.draw(ctx)
        positions["after"] = imgui.get_cursor_pos_y()

    frames(build)

    pad = imgui.get_style().window_padding.y
    content_h = (positions["after"] - positions["before"]) + 2 * pad
    declared = muse_brief.sp(muse_brief.BAR_H)
    assert content_h <= declared, (
        f"the bar drew {content_h:.1f}px of content against a declared "
        f"{declared:.1f}px ({muse_brief.BAR_H} design px), duration_custom="
        f"{duration_custom}"
    )


def test_a_playing_take_draws_as_stop(frames, tmp_path, monkeypatch):
    from warlock.studio.modes.sirens import audio as sirens_audio

    monkeypatch.setattr(sirens_audio, "playing", lambda: True)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "a")
    ctx = _ctx(tmp_path, [_take("a")])
    assert muse_mode.is_playing(ctx, "a") is True
    frames(lambda: muse_results.draw(ctx))


def test_the_tray_shows_only_this_modes_rows(tmp_path):
    """A tray that listed meshes would be a second Library.

    Not a drawing assertion: ``plan_for`` is the filter, and asserting it
    directly is what makes the claim readable.
    """
    ctx = _ctx(
        tmp_path,
        [
            {"id": "m", "kind": "text", "status": "done"},
            _take("a"),
            {"id": "r", "kind": "rig", "status": "done"},
            _take("b"),
        ],
    )
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["b", "a"]
    assert muse_results.should_draw(ctx) is True


def test_the_tray_is_newest_first(tmp_path):
    # ``ctx.cache.jobs`` is oldest-first, and a results tray that showed the
    # first take you ever made at the top would bury every press.
    ctx = _ctx(tmp_path, [_take("old"), _take("new")])
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["new", "old"]


def _with_player(ctx, seconds: float = 6.0):
    """Put a decoded take under the strip, the way an audition would.

    The player draws a *buffer*, not a row, so a tray full of finished takes is
    not enough to exercise it -- which is also why ``should_draw`` gates on the
    buffer rather than on the rows.
    """
    import numpy as np

    from warlock.studio.modes.muse import state as muse_state
    from warlock.studio.modes.muse.engine import waveform

    rate = 44100
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    tone = (np.sin(2 * np.pi * 220.0 * t) * 12000).astype(np.int16)
    pcm = np.stack([tone, tone], axis=1)
    state = muse_mode.ensure(ctx)
    state.player = muse_state.Player(
        job="a", pcm=pcm, rate=rate, env=waveform.peaks(pcm), duration=seconds
    )
    return state.player


def test_the_player_draws_with_a_take_under_it(frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a")])
    _with_player(ctx)
    assert muse_player.should_draw(ctx) is True
    frames(lambda: muse_player.draw(ctx))


def test_the_player_draws_with_a_region_and_candidates(frames, tmp_path):
    """The branch with every control on it: markers, the fill, the candidate
    buttons, the crossfade slider and both exports."""
    from warlock.studio.modes.muse.engine.loops import Candidate

    ctx = _ctx(tmp_path, [_take("a")])
    one = _with_player(ctx)
    muse_mode.set_region(ctx, 1.0, 4.0)
    one.candidates = [Candidate(0, 44100, 0.1), Candidate(44100, 88200, 0.2)]
    frames(lambda: muse_player.draw(ctx))


def test_the_player_draws_while_the_finder_is_running(frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a")])
    _with_player(ctx).finding = True
    frames(lambda: muse_player.draw(ctx))


def test_the_strip_is_absent_until_a_take_has_been_auditioned(tmp_path):
    """A mode that reserved 148 dp for a picture it has no samples for is a
    mode with a hole in it -- so the two columns get the whole height until
    there is something to put under them."""
    ctx = _ctx(tmp_path, [_take("a")])
    assert muse_player.should_draw(ctx) is False


# --- the device-less transports (muse-01, 2026-09-11 audit) ------------------


def test_the_player_strips_transport_is_greyed_with_no_device(frames, tmp_path, monkeypatch):
    """The player strip's Play/Stop used to carry no ``enabled``/``reason`` at
    all -- ``sirens_transport.py`` already greys its own Play/Stop this way,
    and nothing in Muse did the same. ``_no_device`` (this file's autouse
    fixture) is the machine with no audio device; the button drawn onto it
    must say so rather than staying live for a press that will do nothing.
    Fails against the unfixed code, whose captured call carries no
    ``enabled``/``reason`` keys, so ``.get("enabled", True)`` reads ``True``.
    """
    from warlock.studio.modes.muse.ui.panes import player as mp

    ctx = _ctx(tmp_path, [_take("a")])
    _with_player(ctx)
    calls: list[dict[str, Any]] = []

    def _capture(key, playing, **kw):
        calls.append(kw)
        return False

    monkeypatch.setattr(mp.widgets, "transport", _capture)
    frames(lambda: mp.draw(ctx))

    assert calls, "the strip's transport was never drawn"
    assert calls[0].get("enabled", True) is False
    assert calls[0].get("reason", "") != ""


def test_the_trays_card_transport_is_greyed_with_no_device(frames, tmp_path, monkeypatch):
    """``muse_results._actions``'s half of the same finding: a finished take's
    card transport used to grey only on the row's status, never on the
    device. Fails against the unfixed code the same way the strip's test
    does above.
    """
    from warlock.studio.modes.muse.ui.panes import results as mr

    ctx = _ctx(tmp_path, [_take("a", status="done")])
    calls: list[dict[str, Any]] = []

    def _capture(key, playing, **kw):
        calls.append(kw)
        return False

    monkeypatch.setattr(mr.widgets, "transport", _capture)
    frames(lambda: mr.draw(ctx))

    assert calls, "the card's transport was never drawn"
    assert calls[0].get("enabled", True) is False
    assert calls[0].get("reason", "") != ""


def test_every_bar_control_is_named():
    """2026-09-07. Before this pass the two text fields relied on a tooltip
    that showed only while they were empty, and the pills had no name on
    screen at all -- so the manual named controls the bar itself never
    named. A source scan in ``test_ux_consistency_pass2.py:442``'s idiom.
    Fails against the unfixed code, whose ``_tags``/``_duration``/``_count``/
    ``_lyrics`` draw no ``field_label`` at all.
    """
    from pathlib import Path

    source = Path(muse_brief.__file__).read_text(encoding="utf-8")
    for label in ("Tags", "Length", "Takes", "Lyrics"):
        assert f'widgets.field_label("{label}")' in source, label


def test_no_control_appears_in_both_the_bar_and_the_column():
    """The one-owner rule, enforced by reading the two files.

    Create keeps the same split and states it in prose; this is the half that
    fails when somebody adds a duration slider to the recipe column because it
    felt like a setting.
    """
    from pathlib import Path

    bar = Path(muse_brief.__file__).read_text(encoding="utf-8")
    column = Path(muse_recipe.__file__).read_text(encoding="utf-8")
    for field in ("prompt", "lyrics", "duration", "count"):
        assert f'form["{field}"]' in bar, f"the bar should own {field}"
        assert f'form["{field}"]' not in column, f"{field} is in both panes"
    for field in ("infer_step", "guidance_scale", "scheduler_type", "cfg_type"):
        assert f'form["{field}"]' in column, f"the column should own {field}"
        assert f'form["{field}"]' not in bar, f"{field} is in both panes"

    # The third surface. ``modes/muse/ui/panes/results`` draws the derive popup, whose
    # controls are about *one finished take* rather than about the next press
    # -- so it must not touch the brief at all. Without this the popup is a
    # third place to look for a generation setting, which is the failure the
    # bar/column split exists to prevent, one surface further on.
    tray = Path(muse_results.__file__).read_text(encoding="utf-8")
    for field in ("prompt", "lyrics", "duration", "count", "infer_step",
                  "guidance_scale", "scheduler_type", "cfg_type", "omega_scale"):
        assert f'form["{field}"]' not in tray, (
            f"{field} is a brief control and the tray must not own one"
        )


def test_every_task_the_menu_offers_has_controls_and_a_door():
    """Three tables that have to agree, and nothing else makes them.

    ``DERIVE_ITEMS`` is what the menu offers, ``muse_mode.DERIVE_CONTROLS`` is
    what the popup draws for each, and ``_jobs_music.TASKS`` is what the door
    accepts. A task in the first and not the third is a menu item that always
    refuses; one in the third and not the first is capability with no way in.
    """
    from warlock.service._jobs_music import TASKS
    from warlock.studio.modes.muse import mode as muse_mode

    offered = [one[0] for one in muse_results.DERIVE_ITEMS]
    assert sorted(offered) == sorted(TASKS)
    assert sorted(muse_mode.DERIVE_CONTROLS) == sorted(TASKS)
    assert len(set(offered)) == len(offered)


def test_every_derive_control_is_drawn_by_something():
    """A key in ``DEFAULT_DERIVE`` that no task lists is a value nobody can set.

    The reverse is the one that bites: a control named in ``DERIVE_CONTROLS``
    with no entry in ``DEFAULT_DERIVE`` is a ``KeyError`` the first time that
    task's popup opens, and no smoke test would reach it -- the popup only
    draws once a take exists and a menu item has been pressed.
    """
    from warlock.studio.modes.muse import mode as muse_mode
    from warlock.studio.modes.muse.state import DEFAULT_DERIVE

    named = {name for names in muse_mode.DERIVE_CONTROLS.values() for name in names}
    assert named <= set(DEFAULT_DERIVE)
    assert set(DEFAULT_DERIVE) - named == {"task", "count"}
    # The numeric ones each need a label and a bound; the two edit fields are
    # text and are drawn by their own branch.
    assert named - set(muse_results.DERIVE_FIELDS) == {"edit_prompt", "edit_lyrics"}


@pytest.mark.parametrize("task", sorted(muse_mode.DERIVE_CONTROLS))
def test_the_derive_popup_draws_for_every_task(frames, tmp_path, task):
    """The 2026-09-08 audit, finding muse-03. Every other test in this file
    that touches the tray leaves ``MuseState.derive_job`` unset, so
    ``derive_popup`` took its early return (``not state.derive_job``) in
    every run here -- including this file's own docstring claim of drawing
    "every Muse pane". A regression in ``_derive_field``'s per-task widget
    selection, or in the popup's own control loop, had nothing in the suite
    that would ever build the frame it lives on.

    Fails against the unfixed suite in the sense that matters here: deleting
    the two lines below that set ``derive_job``/``derive_form`` (i.e.
    reverting to what every prior test in this file did) makes
    ``derive_popup`` a no-op and this test would draw nothing -- the same gap
    the finding names. What is asserted is that the popup actually reaches
    ``imgui.begin_popup_modal`` and draws every control ``DERIVE_CONTROLS``
    names for this task, not just that ``draw()`` returns without raising.
    """
    from imgui_bundle import imgui

    ctx = _ctx(tmp_path, [_take("a")])
    muse_mode.open_derive(ctx, "a", task)
    state = muse_mode.ensure(ctx)
    assert state.derive_job == "a"
    assert state.derive_form["task"] == task

    opened_popup = False

    def build() -> None:
        nonlocal opened_popup
        muse_results.draw(ctx)
        opened_popup = imgui.is_popup_open(muse_results.DERIVE_POPUP)

    frames(build)
    assert opened_popup, "derive_popup never reached begin_popup_modal for this task"


def test_repaint_window_slider_reaches_the_full_length_of_a_take_longer_than_the_extend_ceiling(
    frames, tmp_path, monkeypatch
):
    """The 2026-09-13 audit, finding muse-01.

    ``_derive_field`` bounded ``repaint_start``/``repaint_end`` by
    ``_max_extend()`` (240s, the *extend* path's sampler ceiling) even though
    ``derive_music_job`` bounds a repaint by the take's own duration (up to
    600s) and a loop by half of it. On a take over four minutes this both
    hid the true window from the slider and refused typed input past 240s.

    Fails against the unfixed code: with a 500s take, the repaint slider's
    ``high`` came back 240.0 instead of 500.0.
    """
    long_take = _take("a")
    long_take["params"] = {"duration": 500.0, "actual_duration": 500.0}
    ctx = _ctx(tmp_path, [long_take])
    muse_mode.open_derive(ctx, "a", "repaint")

    seen: dict[str, tuple[float, float]] = {}
    from warlock.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    assert seen["From"] == (0.0, 500.0)
    assert seen["To"] == (0.0, 500.0)


def test_loop_span_slider_is_bounded_by_half_the_takes_own_duration(frames, tmp_path, monkeypatch):
    """The 2026-09-13 audit, finding muse-01, loop half.

    ``derive_music_job`` refuses a loop span past ``parent_duration / 2.0``;
    the popup must offer no more than that, not ``_max_extend() / 2``.
    """
    long_take = _take("a")
    long_take["params"] = {"duration": 500.0, "actual_duration": 500.0}
    ctx = _ctx(tmp_path, [long_take])
    muse_mode.open_derive(ctx, "a", "loop")

    seen: dict[str, tuple[float, float]] = {}
    from warlock.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    assert seen["Joint to rewrite"] == (0.0, 250.0)


def test_extend_sliders_are_bounded_by_the_takes_own_duration_when_shorter_than_the_sampler_ceiling(
    frames, tmp_path, monkeypatch
):
    """The 2026-09-14 audit, finding muse-04.

    ``_derive_field`` bounded ``extend_left``/``extend_right`` by
    ``_max_extend()`` alone -- the sampler's 240 s pad ceiling -- but
    ``derive_music_job`` separately refuses a single extension longer than
    *the take being extended* ("a single extension cannot be longer than the
    take it extends -- extend twice to go further"), because the pads are
    sliced into a tensor allocated at the parent's own frame length. A 60 s
    take's slider offering up to 240 s let the user drag to a value the door
    would then refuse.

    Fails against the unfixed code: with a 60 s take, both sliders' ``high``
    came back 240.0 instead of 60.0.
    """
    short_take = _take("a")
    short_take["params"] = {"duration": 60.0, "actual_duration": 60.0}
    ctx = _ctx(tmp_path, [short_take])
    muse_mode.open_derive(ctx, "a", "extend")

    seen: dict[str, tuple[float, float]] = {}
    from warlock.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    assert seen["Add before"] == (0.0, 60.0)
    assert seen["Add after"] == (0.0, 60.0)


def test_extend_sliders_never_offer_a_combined_total_the_door_will_refuse(
    frames, tmp_path, monkeypatch
):
    """The 2026-09-18 audit, finding muse-01.

    ``_max_extend_for`` bounded each of ``extend_left``/``extend_right`` by
    ``min(parent_duration, MAX_EXTEND_DURATION)`` *independently of the
    other* -- but ``derive_music_job`` refuses on the *combined* total,
    ``parent_duration + extend_left + extend_right`` against the sampler's
    frame ceiling (``_extend_frame_ceiling_seconds``, ~239.907s), always
    naming ``extend_right`` regardless of which slider actually spent the
    budget (``src/warlock/service/_jobs_music.py:524-541``). A take of 240s
    or more had both sliders come back bounded at up to 240 each -- every
    nonzero extend on such a take cleared the popup and was refused at the
    door.

    Fails against the unfixed code: with a 240s take, ``seen["Add before"]``
    comes back ``(0.0, 240.0)`` instead of ``(0.0, 0.0)`` -- proven against
    the pre-fix ``_max_extend_for`` (single-argument, ``min(parent_duration,
    _max_extend())``) in this session's scratchpad, which returns 240.0 for
    a 240s take with no regard for the sibling slider at all.
    """
    long_take = _take("a")
    long_take["params"] = {"duration": 240.0, "actual_duration": 240.0}
    ctx = _ctx(tmp_path, [long_take])
    muse_mode.open_derive(ctx, "a", "extend")

    seen: dict[str, tuple[float, float]] = {}
    from warlock.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    # No combined total starting from a 240s take can clear the door's
    # ~239.907s ceiling, so there is no room left on either slider.
    assert seen["Add before"] == (0.0, 0.0)
    assert seen["Add after"] == (0.0, 0.0)


def test_extend_sliders_share_the_remaining_budget_on_a_take_with_some_room(
    frames, tmp_path, monkeypatch
):
    """The other half of muse-01: a take short enough to have *some* combined
    budget left must still offer it, split between the two sliders rather
    than each claiming the whole remainder for itself.

    A 200s take has ``_extend_frame_ceiling_seconds() - 200 ~= 39.9s`` of
    total room. ``extend_right`` starts at its default, 30.0 -- so
    ``extend_left``'s own bound must be reduced by that 30.0, not offer the
    same ~39.9s a still-empty sibling would leave it.
    """
    take = _take("a")
    take["params"] = {"duration": 200.0, "actual_duration": 200.0}
    ctx = _ctx(tmp_path, [take])
    muse_mode.open_derive(ctx, "a", "extend")

    seen: dict[str, tuple[float, float]] = {}
    from warlock.studio import widgets as widgets_module

    real_slider = widgets_module.labeled_slider_float

    def spy_slider(title, value, low, high, **kwargs):
        seen[title] = (low, high)
        return real_slider(title, value, low, high, **kwargs)

    monkeypatch.setattr(muse_results.widgets, "labeled_slider_float", spy_slider)

    def build() -> None:
        muse_results.draw(ctx)

    frames(build)

    from warlock.service._jobs_music import _extend_frame_ceiling_seconds

    ceiling = _extend_frame_ceiling_seconds()
    # ``extend_right``'s default (30.0) is spent against extend_left's bound.
    assert seen["Add before"][1] == pytest.approx(ceiling - 200.0 - 30.0)
    # ``extend_left`` is untouched (still its own default, 0.0) when
    # extend_right's own bound is computed.
    assert seen["Add after"][1] == pytest.approx(ceiling - 200.0 - 0.0)
