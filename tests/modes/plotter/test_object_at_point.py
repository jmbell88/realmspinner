"""A point object's hit-test must match where its marker is drawn.

The 2026-09-13 audit (finding plotter-01) found ``_object_at`` testing a point
object against a fixed 8 *map-pixel* box while ``_objects`` draws its marker as
a ring at a fixed ``sp(7)`` **screen** radius -- the same space every other
handle in this file hit-tests in (``_handle_at``, ``_rotate_at``,
``_vertex_at``). Zoomed out, 8 map pixels is a couple of screen pixels and a
click that is visibly on the ring misses; zoomed in, 8 map pixels is far wider
than the drawn ring and a click well away from it grabs the point anyway.
``tests/modes/plotter/_drive.py`` only ever drove zoom 1.0, so this never showed up.
"""

from __future__ import annotations

import pytest

from warlock.studio.shell import paintview

from ._drive import Scene


@pytest.fixture
def scene(monkeypatch):
    return Scene(monkeypatch)


def test_object_at_selects_a_point_object_when_zoomed_out(scene):
    obj = scene.add(kind="point", x=100.0, y=100.0)
    scene.state.selected_objects.clear()
    scene.tab.view.zoom = 0.25

    # The marker's screen position at this zoom, nudged 5 screen pixels away --
    # comfortably inside the drawn ring's sp(7) radius, but 20 map pixels away
    # (5 / 0.25), well outside the old fixed-8-map-pixel box.
    marker = paintview.to_screen(scene.tab.view, (0.0, 0.0), obj.x, obj.y)
    near = (marker[0] + 5.0, marker[1])

    scene.frame(near, click=True)

    assert obj.uid in scene.state.selected_objects, (
        "a click visibly on the marker's ring should select it"
    )
