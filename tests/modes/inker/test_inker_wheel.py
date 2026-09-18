"""The wheel: it writes through the same door the sliders always did, every
colour space is still reachable through it, and its texture does not outlive
the mode.

``tests/modes/inker/test_inker_picker.py`` is the picker's general arithmetic file;
this one is scoped to what the wheel specifically added, so the two can be
told apart by name when either fails.
"""

from __future__ import annotations

import colorsys
import inspect

import pytest

from warlock.studio import tokens
from warlock.studio.modes.inker.ui.panes import picker as inker_picker
from warlock.studio.modes.inker.ui.panes import textures as inker_textures


class _Doc:
    def __init__(self, palette=None, indexed=False):
        self.palette = list(palette or [])
        self.is_indexed = indexed
        self.recoloured: list[tuple[int, tuple]] = []
        self.history = None

    def recolour_slot(self, index, colour):
        self.recoloured.append((index, tuple(colour)))
        self.palette[index] = tuple(colour)
        return True


class _Tab:
    def __init__(self, doc):
        self.doc = doc


class _State:
    def __init__(self, **kwargs):
        self.fg = (255, 255, 255, 255)
        self.bg = (255, 255, 255, 255)
        self.fg_slot = None
        self.picker_target = "fg"
        self.palette_usage = "stale"
        self.picker_space = None
        self.__dict__.update(kwargs)

    def set_fg(self, colour, slot=None):
        self.fg = tuple(int(c) for c in tuple(colour)[:4])
        self.fg_slot = None if slot is None else int(slot)


# --- 1. the wheel writes through write(), for all three targets ---------------

#: A drag straight out to the rim, due "east" -- angle 0, which is pure red at
#: full saturation regardless of radius. Picked because the expected colour
#: needs no trigonometry of its own to state.
_EAST_RIM = {"dx": 40.0, "dy": 0.0, "radius": 40.0}


def test_the_wheel_writes_the_foreground():
    state = _State()
    tab = _Tab(_Doc())
    inker_picker._wheel_pick(None, state, tab, None, (255, 255, 255, 255), **_EAST_RIM)
    assert state.fg == (255, 0, 0, 255)


def test_the_wheel_writes_the_background():
    state = _State(picker_target="bg")
    tab = _Tab(_Doc())
    inker_picker._wheel_pick(None, state, tab, None, (255, 255, 255, 200), **_EAST_RIM)
    # Alpha survives the drag untouched -- the wheel only ever writes hue and
    # saturation; Value and Alpha are their own bars.
    assert state.bg == (255, 0, 0, 200)
    assert state.fg == (255, 255, 255, 255), "the wheel must not also touch the other target"


def test_the_wheel_writes_an_indexed_palette_slot():
    doc = _Doc([(0, 0, 0, 255), (255, 255, 255, 255)], indexed=True)
    tab = _Tab(doc)
    state = _State(fg_slot=1)
    inker_picker._wheel_pick(None, state, tab, 1, (255, 255, 255, 255), **_EAST_RIM)
    assert doc.recoloured == [(1, (255, 0, 0, 255))]
    # Still holding slot 1: the wheel changed what the slot *is*, not which
    # slot the brush came from -- ``write``'s own contract, the sliders had it
    # too.
    assert state.fg_slot == 1
    assert state.fg == (255, 0, 0, 255)


def test_a_drag_past_the_rim_still_writes_the_rim_colour():
    """The clamp lives in ``colorwheel.colour_at``; this is the pane's own
    proof that a drag ten radii out still reaches ``write`` rather than being
    rejected as out of bounds."""
    state = _State()
    tab = _Tab(_Doc())
    inker_picker._wheel_pick(
        None, state, tab, None, (255, 255, 255, 255), dx=400.0, dy=0.0, radius=40.0
    )
    assert state.fg == (255, 0, 0, 255)


# --- 2. every colour space is reachable, and agrees about one colour ----------


def test_every_colour_space_is_reachable_and_agrees_about_one_colour():
    """RGBA, HSV, HSL and Gray are four different readings of the *same*
    colour -- the user's own requirement ("still allow RGBa, HSL, and the
    other color configurations") -- so converting the fixed colour below into
    each space and back must land on it again (RGBA trivially; Gray is a
    projection and is checked against its own definition instead).
    """
    colour = (128, 90, 200, 255)
    state = _State()

    # RGBA: the fields *are* the colour.
    assert colour[:3] == (128, 90, 200)

    # HSV, reconstructed the way ``_hsv_row`` does.
    hue, sat, value = inker_picker._held_hsv(state, colour, inker_picker.HSV_LABELS)
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, sat / 100.0, value / 255.0)
    rebuilt = tuple(inker_picker.clamp8(c * 255.0) for c in (red, green, blue))
    assert rebuilt == colour[:3]

    # HSL, reconstructed the way ``_hsl_row`` does. Held to within a shade
    # rather than exactly, unlike HSV above: the HSL row quantises Saturation
    # and Lightness to 0-100 (matching what the old HSL tab always showed),
    # which is coarser than the 8-bit channel it came from.
    hue2, sat2, light = inker_picker._held_hsv(state, colour, inker_picker.HSL_LABELS)
    red2, green2, blue2 = inker_picker._from_hsl(hue2 / 360.0, sat2 / 100.0, light / 100.0)
    rebuilt2 = tuple(inker_picker.clamp8(c * 255.0) for c in (red2, green2, blue2))
    assert all(abs(a - b) <= 1 for a, b in zip(rebuilt2, colour[:3], strict=True)), rebuilt2

    # Gray: the same weighted luma the palette panel sorts by (LUMA), which is
    # what ``_gray_row`` shows -- not invertible to the colour, so checked
    # against its own formula instead of a round trip.
    level = inker_picker.clamp8(
        sum(weight * channel for weight, channel in zip(inker_picker.LUMA, colour[:3], strict=True))
    )
    assert level == inker_picker.clamp8(0.299 * 128 + 0.587 * 90 + 0.114 * 200)

    # And every one of those four is actually wired to a row the Space combo
    # can select -- not just provable in isolation.
    assert set(inker_picker._ROWS) == {key for key, _ in inker_picker.SPACES}


def test_the_hsv_row_and_the_wheel_read_the_same_held_triple():
    """A hue picked on the wheel and a hue read from the HSV row must be one
    number, not two that happen to agree until one of them moves -- see
    ``_held_hsv``'s docstring (both the wheel and the HSV row pass it
    ``HSV_LABELS``)."""
    state = _State()
    tab = _Tab(_Doc())
    # North -- 90 degrees, full saturation.
    inker_picker._wheel_pick(
        None, state, tab, None, (255, 255, 255, 255), dx=0.0, dy=40.0, radius=40.0
    )
    hue, sat, _value = inker_picker._held_hsv(state, state.fg, inker_picker.HSV_LABELS)
    assert (hue, sat) == (90, 100)


def test_the_held_triple_survives_a_colour_that_has_no_hue_of_its_own():
    """A grey has no hue -- ``colorsys.rgb_to_hsv`` picks 0 arbitrarily -- so a
    caller that re-derived from the RGB every frame would snap the wheel's
    marker to hue 0 the instant Saturation reached zero. ``_held_hsv`` must
    answer with the triple that produced the grey instead, as long as the
    colour has not changed from anywhere else the cache does not know about.
    """
    state = _State()
    grey = (200, 200, 200, 255)
    state.picker_space = (inker_picker.HSV_LABELS, grey[:3], (90, 0, 200))

    hue, sat, value = inker_picker._held_hsv(state, grey, inker_picker.HSV_LABELS)
    assert (hue, sat, value) == (90, 0, 200)

    # A colour the cache does not recognise re-derives instead -- the cache is
    # a record of *this* gesture, not a second opinion about what is selected.
    other = (10, 200, 90, 255)
    reread = inker_picker._held_hsv(state, other, inker_picker.HSV_LABELS)
    assert reread != (90, 0, 200)


# --- 3. the wheel's texture does not outlive the mode --------------------------


class _Texture:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.released = False

    def write(self, data, viewport=None):
        pass

    def release(self):
        self.released = True


class _GL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size, components, data):
        return _Texture(size)


class _Viewer:
    def __init__(self):
        self.ctx = _GL()


class _WheelCtx:
    def __init__(self):
        self.viewer = _Viewer()
        self.state = type("S", (), {"preview": {}})()


def test_the_wheel_texture_is_released_by_the_modes_teardown_sweep():
    """``inker_textures.release_all`` is already wired into the real teardown
    path (``inker_keys.release_all``, and the app-close sweep in ``main.py``);
    the wheel's texture keys under the same ``inker_tex:`` prefix precisely so
    that sweep reaches it with no change to either of those files. If it
    stopped doing that, this is a leaked GL texture every time Inker closes."""
    ctx = _WheelCtx()
    texture = inker_picker._wheel_texture(ctx, 48)
    assert texture is not None
    assert not texture.released
    assert any(key.startswith("inker_tex:") for key in ctx.state.preview)

    inker_textures.release_all(ctx)

    assert texture.released
    assert not any(key.startswith("inker_tex:") for key in ctx.state.preview)


def test_a_ui_scale_change_releases_the_old_wheel_texture_before_rebaking():
    """Keyed on ``tokens.SCALE`` (see ``_wheel_texture``'s docstring) -- a
    session that changes the UI scale must not accumulate one wheel texture
    per scale it has ever been at."""
    ctx = _WheelCtx()
    first = inker_picker._wheel_texture(ctx, 48)
    original_scale = tokens.SCALE
    try:
        tokens.SCALE = original_scale + 0.5
        second = inker_picker._wheel_texture(ctx, 48)
    finally:
        tokens.SCALE = original_scale
    assert first.released
    assert second is not first
    assert not second.released


# --- 4. the wheel is addressable by the control probe --------------------------


@pytest.fixture
def frames():
    """A bare imgui context, built and destroyed around this file.

    ``tests/modes/sirens/test_sirens_panes_smoke.py``'s own fixture of the same name and
    for the same reason: at most one imgui context may exist at a time, and a
    file that wants one builds and destroys it rather than relying on
    collection order. No GL: ``renderer_has_textures`` is what lets imgui
    finish a frame without a backend to claim its font atlas, and the wheel
    draws no image when ``ctx.viewer`` is ``None`` -- exactly the state every
    other headless pane test in this suite runs in.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    yield imgui
    imgui.destroy_context()


def test_the_wheel_is_reachable_by_the_control_probe(frames, monkeypatch):
    """``probe`` is what ``dev/scripts/exercise_mode.py`` and the driver in
    ``tests/studio/test_probe.py``-shaped suites use to find and click a control; a
    canvas drawn straight to the draw list is invisible to it unless it calls
    ``controls._finish_item`` itself (see ``_wheel``'s call to it, right after
    the invisible button). Env-gated normally (``probe.ENABLED``), so this
    turns the gate on for the one frame that needs it.
    """
    from warlock.studio import probe

    monkeypatch.setattr(probe, "ENABLED", True)
    probe.begin_frame()

    state = _State()
    tab = _Tab(_Doc())
    ctx = type("Ctx", (), {"viewer": None, "state": type("S", (), {"preview": {}})()})()

    frames.new_frame()
    frames.set_next_window_size((320.0, 400.0))
    frames.begin("host")
    inker_picker._wheel(ctx, state, tab, None, (255, 255, 255, 255))
    frames.end()
    frames.render()

    found = [control for control in probe.census() if control.kind == "colorwheel"]
    assert len(found) == 1
    wheel = found[0]
    assert wheel.rect[2] > 0.0 and wheel.rect[3] > 0.0, "a driver cannot click a zero-size rect"
    assert wheel.visible


def test_a_pick_on_the_wheel_at_value_zero_returns_a_colour_not_black():
    """Otherwise the disc reads as broken on the very first click.

    Inker opens on a black foreground, and every colour at Value 0 is black --
    so a pick that honoured the value would multiply the chosen hue by zero and
    write back the black already in hand. The user clicks a magenta, the swatch
    does not move, and the control has told them it does not work. Nothing is
    discarded by lifting it: a value of 0 carries neither hue nor saturation,
    and the Value bar sits directly beneath the wheel to take it back down.
    """
    state = _State()
    tab = _Tab(_Doc())
    state.fg = (0, 0, 0, 255)
    inker_picker._wheel_pick(None, state, tab, None, (0, 0, 0, 255), **_EAST_RIM)
    assert state.fg == (255, 0, 0, 255), "a pick at Value 0 wrote black back"


def test_the_disc_is_never_drawn_darker_than_its_floor():
    """A literal value tint draws a **black circle**, which hides the control.

    The tint is honest -- at Value 0 the wheel would otherwise promise a bright
    colour the picks cannot produce -- but taken to zero it removes the one
    thing that says "this is a colour wheel" from a pane a new user meets on
    their first drawing. The floor keeps every hue legible while still visibly
    darkening as the bar comes down.
    """
    assert 0 < inker_picker.WHEEL_DIM_FLOOR < 255
    source = inspect.getsource(inker_picker._wheel)
    assert "WHEEL_DIM_FLOOR" in source, "the tint no longer respects the floor"
