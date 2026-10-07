"""The viewport's two pieces of chrome, asserted without a viewport.

Both are pure, and that is what this file is for: "does the +X ball sit on the
right when the camera is at the front" and "does edge mode mention the loop
shortcut" are questions a headless test can ask and a screenshot cannot be made
to fail on.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.viewer.camera import Camera

SIZE = 84.0
CENTRE = SIZE * 0.5


def _at(name: str) -> list[clay_hints.AxisBall]:
    camera = Camera()
    assert camera.look_along(name)
    return clay_hints.axis_layout(camera.view(), SIZE)


# --- the navigation widget ---------------------------------------------------


def test_there_are_six_ends_and_three_of_them_are_lettered():
    balls = _at("front")
    assert len(balls) == 6
    assert sorted(ball.label for ball in balls if ball.label) == ["X", "Y", "Z"]
    assert all(ball.positive for ball in balls if ball.label), (
        "only the positive end of an axis carries its letter"
    )


@pytest.mark.parametrize("name", sorted(Camera.AXIS_VIEWS))
def test_every_ball_puts_the_camera_where_its_axis_points(name):
    """The pairing of a ball to a ``Camera.AXIS_VIEWS`` name, checked against
    the camera rather than against the table that declares it.

    Looking *from* the +Z side is what "front" means, so with the camera there
    the +Z ball must be the one nearest the viewer -- and dead centre, because
    an axis pointing straight at the eye projects to a point.
    """
    balls = _at(name)
    nearest = balls[-1]
    assert nearest.view == name
    assert nearest.x == pytest.approx(CENTRE, abs=0.5)
    assert nearest.y == pytest.approx(CENTRE, abs=0.5)
    assert nearest.depth == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", sorted(Camera.AXIS_VIEWS))
def test_the_opposite_ball_is_hidden_behind_it(name):
    """They land on the same pixel, and which is on top is the whole of what
    tells the reader which way they are looking. Back to front, so the far one
    is drawn first."""
    balls = _at(name)
    assert balls[0].depth == pytest.approx(-1.0, abs=1e-6)
    assert balls[0].x == pytest.approx(balls[-1].x, abs=0.5)


def test_the_balls_are_ordered_back_to_front():
    balls = clay_hints.axis_layout(Camera().view(), SIZE)
    assert [ball.depth for ball in balls] == sorted(ball.depth for ball in balls)


def test_x_is_to_the_right_and_y_is_up_at_a_front_on_camera():
    """Screen y grows downward, which is the one conversion the layout owes and
    the easiest to get backwards."""
    balls = {ball.view: ball for ball in _at("front")}
    assert balls["right"].x > CENTRE
    assert balls["left"].x < CENTRE
    assert balls["top"].y < CENTRE, "up the screen is a smaller y"
    assert balls["bottom"].y > CENTRE


def test_every_ball_stays_inside_the_box():
    for name in Camera.AXIS_VIEWS:
        for ball in _at(name):
            assert 0.0 <= ball.x <= SIZE, (name, ball)
            assert 0.0 <= ball.y <= SIZE, (name, ball)


def test_the_layout_scales_with_the_box():
    small = clay_hints.axis_layout(Camera().view(), 40.0)
    big = clay_hints.axis_layout(Camera().view(), 80.0)
    for a, b in zip(small, big, strict=True):
        assert b.x == pytest.approx(a.x * 2.0)
        assert b.y == pytest.approx(a.y * 2.0)


def test_a_degenerate_matrix_still_produces_six_balls():
    """A pane must not raise on the first frame, before a camera exists."""
    assert len(clay_hints.axis_layout(np.eye(4), SIZE)) == 6


# --- the hint line -----------------------------------------------------------


@pytest.mark.parametrize("mode", clay_ops.ALL_MODES)
@pytest.mark.parametrize("tool", [key for key, _label, _key in clay_state.TOOLS])
def test_every_mode_and_tool_pair_has_a_hint(mode, tool):
    line = clay_hints.hint(mode, tool)
    assert line and "  " not in line
    # The two navigation buttons are on every line: they are what a newcomer to
    # a 3D viewport asks about first and what a manual is least open at.
    assert "Alt+LMB orbit" in line and "MMB pan" in line


def test_the_element_modes_advertise_only_the_verbs_they_still_have():
    """Alt+click loop/ring and grow/shrink went with the loop and ring queries:
    a hint that names a chord nothing listens to is offer-then-refuse."""
    for mode in ("vertex", "edge", "face"):
        line = clay_hints.hint(mode, "select")
        assert "marquee" in line and "L linked" in line
        assert "Alt+click" not in line and "grow/shrink" not in line
    assert clay_hints.hint("nonsense", "select") == clay_hints.hint("object", "select")


# --- the measurement line ----------------------------------------------------


def test_a_live_measurement_outranks_the_ordinary_legend():
    assert clay_hints.resolve_hint(measure="area  1.0000 m²", default="x") == "area  1.0000 m²"
    assert clay_hints.resolve_hint(measure="", default="x") == "x"


# --- the keys the line names -------------------------------------------------


def test_keys_named_finds_the_chords_and_the_bare_letters():
    found = clay_hints.keys_named(clay_hints.hint("edge", "move"))
    assert {"L", "G", "4", "LMB", "MMB", "Alt+LMB"} <= found


def test_keys_named_does_not_read_english_as_a_binding():
    """"drag a ring" -- the article is not the A key. A single letter counts
    only when it is capital, which is how this app writes every binding."""
    assert clay_hints.keys_named("R rotate . drag a ring") == {"R"}
    assert "X" in clay_hints.keys_named("X/Y/Z lock")
    assert clay_hints.keys_named("grow/shrink the selection") == set()


def test_every_key_the_line_names_is_a_key_the_mode_listens_to():
    """The parity that matters: a hint naming a binding nothing implements is
    worse than no hint, because it is read as a promise."""
    from realmspinner.studio.modes.clay import mode as clay_mode

    letters = set()
    for mode in clay_ops.ALL_MODES:
        for tool, _label, _key in clay_state.TOOLS:
            letters |= {
                key.lower()
                for key in clay_hints.keys_named(clay_hints.hint(mode, tool))
                if len(key) == 1 and key.isupper()
            }
    known = set(clay_mode.TOOL_KEYS) | {"g", "s", "r", "l"}
    assert letters <= known, sorted(letters - known)


def test_the_hint_line_names_no_multi_character_binding_nothing_implements():
    """clay-05 (2026-09-18 audit): the line said "Tab edit"/"Tab object", and
    ``mode.py``'s own comment on ``ELEMENT_KEYS`` says Tab is deliberately
    unbound -- imgui's keyboard navigation owns it, and binding it too would
    move focus out of the viewport as well as changing the mode. The parity
    test above never caught it because it only ever looked at single letters;
    this widens the same check to every named token -- "Tab", "Enter", digit
    groups, all of it -- against the keys Clay actually listens to.

    Fails against the unfixed code: ``keys_named`` found "Tab" in the object
    and every element mode's line, and "Tab" is not a key ``mode.handle_key``
    binds anything to.
    """
    from realmspinner.studio.modes.clay import mode as clay_mode

    # The named tokens Clay's own key handler answers to (the mouse buttons and
    # the wheel are always true; a bare press never commits or cancels, so
    # "Enter"/"Esc" have no place on this line -- the drag has its own, see
    # ``drag_readout``).
    known_named = {"LMB", "MMB", "RMB", "wheel"}
    known_digits = set(clay_mode.ELEMENT_KEYS)

    for mode in clay_ops.ALL_MODES:
        for tool, _label, _key in clay_state.TOOLS:
            found = clay_hints.keys_named(clay_hints.hint(mode, tool))
            multi = {key for key in found if len(key) > 1 and "+" not in key}
            digits = {key for key in found if key.isdigit()}
            assert multi <= known_named, (mode, tool, sorted(multi - known_named))
            assert digits <= known_digits, (mode, tool, sorted(digits - known_digits))


def test_measure_lines_own_import_matches_what_the_module_docstring_now_claims():
    """The 2026-09-26 audit's clay-view-06: this module's own docstring used
    to say "Nothing here imports outward" outright, but ``measure_line``
    reaches into ``kernels.mesh.measure`` for its arithmetic -- a real
    outward edge, just a local one rather than a module-scope import. No
    import-pin test covers this module the way ``tests/_pure_packages.py``
    covers the kernel packages and each mode's ``engine/`` (this module was
    never one of those), so the fix is to make the docstring's claim true
    rather than to add a pin nothing else here follows.

    Fails against the unfixed docstring: it claimed no outward imports at
    all, with nothing narrowing that to module scope, while this very
    import exists a few dozen lines below it.
    """
    import inspect

    source = inspect.getsource(clay_hints._compute_measure_line)
    assert "from ..kernels.mesh import measure" in source, (
        "measure_line's own arithmetic must still reach kernels.mesh.measure"
    )
    assert "module scope" in (clay_hints.__doc__ or ""), (
        "the module docstring must narrow its outward-import claim to module scope, "
        "since a function-local import is a real outward edge this docstring used to deny"
    )
