"""Second-run 2026-09-23 audit fixes for Plotter.

plotter-01: a polyline object's hit-test must match where the line is drawn,
the same fault the 2026-09-13 audit (plotter-01) fixed for point objects
(``tests/modes/plotter/test_object_at_point.py``) but left in place for
``_near_polyline``, which compared against a fixed 8 *map-pixel* tolerance
instead of a screen-space one. Zoomed out, a click visibly on the drawn line
missed it.
"""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.plotter.engine._map_model import Polyline
from realmspinner.studio.shell import paintview

from ._drive import Scene


@pytest.fixture
def scene(monkeypatch):
    return Scene(monkeypatch)


def test_object_at_selects_a_polyline_object_when_zoomed_out(scene):
    obj = scene.add(
        x=100.0,
        y=100.0,
        shape=Polyline(((0.0, 0.0), (100.0, 0.0))),
    )
    scene.state.selected_objects.clear()
    scene.tab.view.zoom = 0.25

    # A point on the line's midpoint, nudged 5 screen pixels perpendicular --
    # comfortably inside the sp(7) screen tolerance every other handle in this
    # file hit-tests against, but 20 map pixels away (5 / 0.25), well outside
    # the old fixed-8-map-pixel box.
    midpoint = paintview.to_screen(scene.tab.view, (0.0, 0.0), obj.x + 50.0, obj.y)
    near = (midpoint[0], midpoint[1] + 5.0)

    scene.frame(near, click=True)

    assert obj.uid in scene.state.selected_objects, (
        "a click visibly on the drawn line should select it"
    )
