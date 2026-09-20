"""The camera a character sheet is framed from: one table, one default.

A preset is a *name for an elevation*, and the whole risk in offering names is
that the name and the number stop agreeing -- a combo saying "Isometric" over a
sheet rendered at 35 degrees is worse than no combo at all. So the table has one
home, ``kernels.charsheet.CAMERA_PRESETS``: the door reads it, the form reads
the door, and the worker matches against it when it stamps the sidecar.

P9 (2026-09-18): this test followed Troupe's own ``ui/panes/settings.py`` into
Poser's ``ui/panes/sheet.py`` when that mode folded in. The service-side
assertions (``_q_troupe``, ``service.troupe``) are unchanged -- neither module
moved.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from realmspinner import _q_troupe
from realmspinner.kernels import charsheet
from realmspinner.kernels import sheet as sheetlib
from realmspinner.service import troupe as svc_troupe
from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet


def test_the_form_and_the_door_read_one_preset_table(svc):
    """The combo's choices are ``troupe_options``' answer and nothing else, and
    ``troupe_options`` is ``charsheet.CAMERA_PRESETS`` reshaped.

    Pinned by scanning as well as by comparing, because the failure this is
    written against is not a wrong answer today -- it is a copy of the angles
    landing in the pane or the door tomorrow and drifting from the table the
    renderer frames from.
    """
    options = svc_troupe.troupe_options(svc)
    presets = options["camera_presets"]
    assert list(presets) == [key for key, _label, _angle in charsheet.CAMERA_PRESETS]
    assert presets == {
        key: {"label": label, "elevation": angle}
        for key, label, angle in charsheet.CAMERA_PRESETS
    }

    # The pane builds its choices out of ``options["camera_presets"]`` -- it
    # never names a preset or an angle itself. Scanned as the specific
    # functions that draw the camera/layout/style form fields, not the whole
    # module: P9 (2026-09-18) merged Troupe's four panes into one file, and
    # the sheet preview's own drawing (``_scorecard``, unrelated ``0.0``
    # defaults and the like) would otherwise collide with this scan by
    # coincidence rather than by actually naming a preset.
    pane = "".join(
        inspect.getsource(fn)
        for fn in (
            poser_sheet._size,
            poser_sheet._camera_helper,
            poser_sheet._layout,
            poser_sheet._style,
            poser_sheet._frame_rate,
            poser_sheet._logical_size,
            poser_sheet._palette,
            poser_sheet.draw_new_character,
            poser_sheet._build_another,
            poser_sheet._submit_new_character,
        )
    )
    assert '"camera_presets"' in pane
    for key, label, angle in charsheet.CAMERA_PRESETS:
        # As a *literal*: "side" is a substring of "beside", which this file's
        # prose says several times about controls that sit next to each other.
        assert f'"{key}"' not in pane, f"the form names the preset {key!r} itself"
        assert f"'{key}'" not in pane, f"the form names the preset {key!r} itself"
        assert repr(angle) not in pane, f"the form names {angle} itself"
        assert f'"{label}"' not in pane, f"the form names the label {label!r} itself"

    # And across the whole package, two modules spell the default key -- and
    # only two.
    #
    # ``charsheet.py`` owns it. ``resolve.py`` names it as the *target of a
    # vocabulary*: the resolver's job is to turn "3/4 top down" into a preset
    # key, so the key has to be written down on the right-hand side of that
    # table, and there is no arithmetic there to derive it from. It is allowed
    # here on one condition, which the next assertion enforces -- **the
    # resolver may spell a preset's key and may never spell its angle.** A key
    # that drifts is caught the moment it drifts, by the two-way pin in
    # ``tests/characters/test_resolve.py`` (every key the vocabulary emits is a
    # real preset, *and* every preset is askable in words). An angle that
    # drifted would be silent, which is the failure this whole test exists for.
    root = Path(charsheet.__file__).resolve().parents[1]
    homes = sorted(
        path.name
        for path in root.rglob("*.py")
        if charsheet.DEFAULT_CAMERA_PRESET in path.read_text(encoding="utf-8")
    )
    assert homes == ["charsheet.py", "resolve.py"], homes

    resolver = (root / "characters" / "resolve.py").read_text(encoding="utf-8")
    for _key, _label, angle in charsheet.CAMERA_PRESETS:
        assert repr(angle) not in resolver, f"the resolver names the angle {angle}"


def test_the_default_preset_is_three_quarter_top_down_at_35_degrees():
    """The angle nearly every 2D game with depth is drawn at, and the reason it
    is the default rather than ``isometric``: 30 degrees is what every sheet
    this program has rendered so far used, and it was a renderer default that
    nobody chose."""
    assert charsheet.DEFAULT_CAMERA_PRESET == "three_quarter_top_down"
    angles = {key: angle for key, _label, angle in charsheet.CAMERA_PRESETS}
    assert angles[charsheet.DEFAULT_CAMERA_PRESET] == 35.0
    # The one preset that is a number the program already had, named.
    assert angles["isometric"] == sheetlib.DEFAULT_ELEVATION


def test_a_custom_elevation_writes_a_null_preset():
    """A sheet framed at an angle off the ladder records the angle and no name.

    The sidecar must not claim a framing the render was not made at, so the
    preset is matched exactly rather than snapped to the nearest -- 34 degrees
    is not "3/4 top-down", it is 34 degrees.
    """
    custom = _q_troupe._camera_meta(34.0, pixel_size=32, margin=sheetlib.FRAME_MARGIN)
    assert custom["preset"] is None
    assert custom["elevation"] == 34.0
    assert custom["pixel_size"] == 32
    assert custom["render_size"] == charsheet.RENDER_SIZE
    assert custom["projection"] == "orthographic"
    assert custom["frame_margin"] == sheetlib.FRAME_MARGIN

    # And a sheet framed *on* the ladder names the preset it was framed from.
    for key, _label, angle in charsheet.CAMERA_PRESETS:
        named = _q_troupe._camera_meta(
            angle, pixel_size=32, margin=sheetlib.FRAME_MARGIN
        )
        assert named["preset"] == key
        assert named["elevation"] == angle

    # The worker stamps it on every sheet, beside the layout snapshot.
    source = inspect.getsource(_q_troupe.TroupeOps._charsheet)
    assert 'meta["camera"] = _camera_meta(' in source


# --- camera_line --------------------------------------------------------
#
# The 2026-09-20 audit, finding troupe-04: a pure, branching string builder
# with no test of its own, unlike its directly-tested neighbours in the same
# file (``_pixel_report_lines``). Every branch below.


def test_camera_line_is_empty_with_no_camera_and_no_elevation():
    assert poser_sheet.camera_line({}) == ""


def test_camera_line_falls_back_to_elevation_with_no_camera_block():
    """A sheet old enough to predate ``camera`` still says what it was framed
    at, off the bare top-level ``elevation`` key."""
    assert poser_sheet.camera_line({"elevation": 34.0}) == "framed at 34 degrees"


def test_camera_line_names_a_known_preset():
    record = {"camera": {"preset": "isometric", "elevation": 30.0}}
    assert poser_sheet.camera_line(record) == "Isometric, 30 degrees"


def test_camera_line_calls_an_unknown_preset_custom():
    record = {"camera": {"preset": "", "elevation": 34.0}}
    assert poser_sheet.camera_line(record) == "custom, 34 degrees"


def test_camera_line_defaults_a_missing_elevation_to_zero():
    record = {"camera": {"preset": "side"}}
    assert poser_sheet.camera_line(record) == "Side, 0 degrees"


def test_camera_line_appends_front_yaw_when_present():
    record = {"camera": {"preset": "side", "elevation": 0.0, "front_yaw": 90.0}}
    assert poser_sheet.camera_line(record) == (
        "Side, 0 degrees -- front at 90 degrees"
    )


def test_camera_line_appends_projection_when_set():
    record = {
        "camera": {"preset": "top_down", "elevation": 60.0, "projection": "orthographic"}
    }
    assert poser_sheet.camera_line(record) == (
        "Top-down, 60 degrees -- orthographic"
    )


def test_camera_line_appends_pixel_size_when_set():
    record = {
        "camera": {
            "preset": "three_quarter_top_down", "elevation": 35.0, "pixel_size": 32,
        }
    }
    assert poser_sheet.camera_line(record) == (
        "3/4 top-down, 35 degrees -- 32 px sprite"
    )


def test_camera_line_joins_every_optional_part_in_order():
    record = {
        "camera": {
            "preset": "isometric",
            "elevation": 30.0,
            "front_yaw": 180.0,
            "projection": "orthographic",
            "pixel_size": 64,
        }
    }
    assert poser_sheet.camera_line(record) == (
        "Isometric, 30 degrees -- front at 180 degrees -- orthographic "
        "-- 64 px sprite"
    )
