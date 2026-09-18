"""Every Sirens pane, drawn -- on a machine with no GPU and no sound card.

The other pane smoke tests in this suite build a *renderer* over a real GL
context and skip where there is none, which is most CI and every remote shell.
That skip is what let a real bug ship: ``sirens_patterns`` drew its caret with
``add_rect``'s thickness and flags the wrong way round, which type-errors on
every frame that has a grid on screen -- and nothing here drew a pane, so
nothing noticed.

Nothing in this file presents anything, so no GL is needed. imgui hands its font
atlas to a backend and will not finish a frame until one claims it; declaring
``renderer_has_textures`` is that claim, and with nothing to draw into it costs
a texture upload that never happens. What the frame still does is run every
widget call, every draw-list call and every layout pass for real, which is
exactly the layer where a wrong argument order lives.

The panes are drawn into a window with a **stated size**. That is not cosmetic:
the grid skips a channel whose column starts past the content region, so a
default-sized window drew nothing at all and passed while the bug was there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio.modes.sirens import mode as sirens_mode
from warlock.studio.modes.sirens.engine import document as D
from warlock.studio.modes.sirens.engine import instruments as inst
from warlock.studio.modes.sirens.ui.panes import bridge as sirens_bridge
from warlock.studio.modes.sirens.ui.panes import effects as sirens_effects
from warlock.studio.modes.sirens.ui.panes import envelopes as sirens_envelopes
from warlock.studio.modes.sirens.ui.panes import instruments as sirens_instruments
from warlock.studio.modes.sirens.ui.panes import orders as sirens_orders
from warlock.studio.modes.sirens.ui.panes import patterns as sirens_patterns
from warlock.studio.modes.sirens.ui.panes import transport as sirens_transport

from .test_sirens_mode import FakeCtx, _tab

PANES = (
    ("sirens-patterns", sirens_patterns),
    ("sirens-transport", sirens_transport),
    ("sirens-orders", sirens_orders),
    ("sirens-instruments", sirens_instruments),
    ("sirens-envelopes", sirens_envelopes),
    ("sirens-effects", sirens_effects),
    ("sirens-bridge", sirens_bridge),
)

#: Wide and tall enough that the grid draws every channel and the envelope
#: editor draws four graphs rather than four lines. A sidebar is 300 design
#: pixels and the centre column is the rest of the window.
WINDOW = (760.0, 900.0)

#: A single sidebar at its own design width (``skeletons.COLUMN_W``), used by
#: the narrow-window case below. Every right-column pane draws inside exactly
#: this width in the running app, so a control that only fits at 760 px was
#: never actually exercised at the size it is really drawn at.
NARROW_WINDOW = (300.0, 900.0)


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


def _loaded(ctx: FakeCtx) -> Any:
    """A song with something in every pane: notes, a selection, a sample, and
    an instrument whose four sequences are all non-empty."""
    tab = _tab(ctx)
    doc = tab.doc
    pattern = doc.patterns[0]
    for row, note in enumerate((48, 52, 55)):
        doc.set_cell(pattern.uid, row * 4, 0, D.NOTE, note)
        doc.set_cell(pattern.uid, row * 4, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.update_instrument(
        doc.instruments[0].uid,
        volume=inst.Sequence(values=(15, 12, 8, 4, 0), loop=0, release=2),
        arpeggio=inst.Sequence(values=(0, 4, 7), loop=0),
        pitch=inst.Sequence(values=(-40, 0, 40)),
        duty=inst.Sequence(values=(0, 1, 2, 3), loop=0),
    )
    effect = doc.add_oneshot("coin", rows=2)
    doc.set_cell(effect.pattern, 0, 0, D.NOTE, 60)
    doc.set_cell(effect.pattern, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.set_sample("kick", np.zeros(128, dtype=np.float32))
    sample = next(one for one in doc.instruments if one.kind == "sample")
    doc.update_instrument(sample.uid, sample="kick")
    state = sirens_mode.ensure(ctx)
    state.oneshot = effect.uid
    state.anchor = (0, 0)
    state.row, state.channel = 4, 1
    return tab


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_a_song_open(name, pane, frames):
    ctx = FakeCtx()
    _loaded(ctx)
    frames(lambda: pane.draw(ctx))


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_nothing_open(name, pane, frames):
    """The empty state is a frame too, and it is the first one a user sees."""
    ctx = FakeCtx()
    frames(lambda: pane.draw(ctx))


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_in_a_narrow_sidebar(name, pane, frames):
    """K97's window, for Sirens (the 2026-09-07 audit).

    Every right-column pane in the running app draws inside a 300 px sidebar,
    not the 760 px this file otherwise uses -- so a label row built out of an
    unmeasured run of ``same_line`` calls (the envelope header's two small
    buttons, before ``same_line_or_wrap``) could clip past the right edge on
    every real launch and this suite would still be green, because nothing
    here ever asked imgui to lay the row out at the width it is actually drawn
    at. This does not read pixels back -- imgui gives no headless way to do
    that -- but it does run the same wrapping arithmetic
    ``same_line_or_wrap``/``button_width`` exercise at the narrow width,
    which a 760 px frame never reaches.
    """
    ctx = FakeCtx()
    _loaded(ctx)
    frames(lambda: pane.draw(ctx), size=NARROW_WINDOW)


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_at_a_larger_ui_scale(name, pane, frames):
    """The other half of K97's reasoning: a hardcoded pixel size is invisible
    at the one scale this suite otherwise runs at (1.0) and wrong at every
    other. ``grid_width``, ``button_width`` and ``same_line_or_wrap`` all ask
    ``tokens.sp``/the live imgui style rather than a literal, which is exactly
    what stops a wider frame padding at 1.6x from pushing a row's last button
    past the content region the way a bare ``same_line()`` used to.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme, tokens

    ctx = FakeCtx()
    _loaded(ctx)
    tokens.set_scale(1.6)
    # ``theme.apply`` bakes ``item_spacing``/``frame_padding`` into the style
    # at call time (its own docstring: "Called once, after the context and
    # fonts exist") -- ``grid_width`` and ``button_width`` read those from the
    # *style*, not from ``tokens.SCALE`` directly, so a bare ``set_scale``
    # with no second ``apply`` would leave them at 1.0x and this test would
    # exercise nothing a 1.0x frame does not already.
    theme.apply(imgui)
    try:
        frames(lambda: pane.draw(ctx))
    finally:
        tokens.set_scale(1.0)
        theme.apply(imgui)


def test_the_grid_draws_its_caret(frames):
    """The bug this file was written for. The caret rectangle is drawn only
    where the caret is, so a window too small to reach that column passed while
    the call itself could not run."""
    ctx = FakeCtx()
    _loaded(ctx)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=0)
    frames(lambda: sirens_patterns.draw(ctx))


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_the_grid_draws_a_caret_in_every_column(column, frames):
    """Every column takes keys now, so every column is somewhere the caret can
    stop -- and the caret is drawn from the cell's *text*, whose width is a
    different number in each of the five."""
    ctx = FakeCtx()
    _loaded(ctx)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=column)
    frames(lambda: sirens_patterns.draw(ctx))


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_the_grid_draws_a_caret_over_a_half_typed_nibble(column, frames):
    """Mid-entry the caret rings one character rather than the cell, which is
    the only thing on screen that says a second keystroke is still owed. Drawn
    for every column, because ``state.digit`` is one number and an empty cell's
    text is shorter than a full one's."""
    ctx = FakeCtx()
    _loaded(ctx)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=column)
    sirens_mode.ensure(ctx).digit = 1
    frames(lambda: sirens_patterns.draw(ctx))


def test_the_envelope_editor_draws_an_instrument_with_no_sequences(frames):
    """``default`` gives a new instrument a volume curve and nothing else, and
    three of the four graphs are then empty -- which is the ordinary case, not
    an edge one."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    bare = tab.doc.add_instrument(kind="triangle")
    tab.doc.update_instrument(bare.uid, volume=inst.Sequence())
    sirens_mode.ensure(ctx).instrument = bare.uid
    frames(lambda: sirens_envelopes.draw(ctx))


def test_the_envelope_editor_draws_a_sequence_at_the_engines_ceiling(frames):
    """256 steps is the widest a graph is ever asked to draw, and the column
    width it works out to is a fraction of a pixel."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.instruments[0].uid
    tab.doc.update_instrument(
        uid,
        volume=inst.Sequence(
            values=tuple(range(inst.MAX_SEQUENCE_LEN)), loop=8, release=200
        ),
    )
    sirens_mode.ensure(ctx).instrument = uid
    frames(lambda: sirens_envelopes.draw(ctx))


def test_the_effects_pane_draws_an_effect_whose_pattern_is_gone(frames):
    """Unreachable through the app -- ``add_oneshot`` mints the pattern and the
    pair is one undo step -- but a hand-edited ``.wsng`` can carry it, and a row
    that renders as an exception is worse than one that says what is wrong."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    one = tab.doc.add_oneshot("orphan", rows=2)
    tab.doc.remove_pattern(one.pattern)
    sirens_mode.ensure(ctx).oneshot = one.uid
    frames(lambda: sirens_effects.draw(ctx))


# --- a click picks the column ---------------------------------------------------
#
# "Click on ``Fxx``, type, get a note": the press moved the row and the channel
# and left the column where it was (the 2026-09-02 review, section 8).


def test_a_click_inside_a_column_picks_that_column():
    from warlock.studio.modes.sirens.ui.panes.patterns import column_at

    widths = [30.0, 20.0, 20.0, 10.0, 20.0]
    gap = 6.0
    assert column_at(0.0, widths, gap) == 0
    assert column_at(29.0, widths, gap) == 0
    assert column_at(40.0, widths, gap) == 1
    assert column_at(70.0, widths, gap) == 2
    assert column_at(96.0, widths, gap) == 3
    assert column_at(110.0, widths, gap) == 4


def test_the_gap_after_a_column_belongs_to_it():
    """A caret that refused to move because the press landed one pixel wide of
    a glyph is a control that works most of the time."""
    from warlock.studio.modes.sirens.ui.panes.patterns import column_at

    widths = [30.0, 20.0, 20.0, 10.0, 20.0]
    assert column_at(33.0, widths, 6.0) == 0
    assert column_at(35.9, widths, 6.0) == 0
    assert column_at(36.0, widths, 6.0) == 1


def test_a_click_past_the_last_column_clamps_rather_than_refusing():
    from warlock.studio.modes.sirens.ui.panes.patterns import column_at

    widths = [30.0, 20.0, 20.0, 10.0, 20.0]
    assert column_at(10_000.0, widths, 6.0) == 4
    assert column_at(-5.0, widths, 6.0) == 0


def test_retarget_popup_refuses_a_selection_while_the_song_is_busy(frames, monkeypatch):
    """Finding sirens-04, the 2026-09-13 audit. The order list's "point this
    entry at another pattern" popup drew its rows with no regard for
    ``editable``, so a popup left open across a save starting still called
    ``set_order`` on a tab the rest of the pane was refusing to touch --
    imgui's own disabled state does not close an already-open popup, and
    imgui's real click-blocking under ``BeginDisabled`` is not something this
    suite has a headless way to drive. Reproduced against the unfixed code by
    faking a click through ``controls.selectable`` regardless of ``enabled``:
    the unfixed ``_retarget`` applied it whether or not the row could really
    have been clicked.
    """
    from imgui_bundle import imgui

    from warlock.studio.modes.sirens.ui.panes import orders as sirens_orders

    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    doc.set_order([first])

    # Simulates a stale click landing on the popup's first row, regardless of
    # whether the row was actually clickable.
    monkeypatch.setattr(
        sirens_orders.controls, "selectable", lambda *a, **k: (True, False)
    )
    monkeypatch.setattr(imgui, "begin_popup", lambda *_a, **_k: True)
    monkeypatch.setattr(imgui, "end_popup", lambda: None)
    monkeypatch.setattr(imgui, "open_popup", lambda *_a, **_k: None)

    # ``frames`` runs ``build`` for its side effects and returns nothing, so
    # ``_retarget``'s answer is captured through this dict rather than a
    # return value.
    result: dict[str, bool] = {}

    def busy_call():
        result["changed"] = sirens_orders._retarget(ctx, tab, 0, first, False)

    frames(busy_call)
    assert not result["changed"]
    assert list(doc.order) == [first]

    def editable_call():
        result["changed"] = sirens_orders._retarget(ctx, tab, 0, first, True)

    frames(editable_call)
    assert result["changed"]
    assert list(doc.order) == [second]
