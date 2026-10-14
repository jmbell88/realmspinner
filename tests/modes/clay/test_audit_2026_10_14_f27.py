"""The 2026-10-14 audit: Clay viewport findings clay-51 (clay-view-06), clay-49 (clay-view-04)."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import timeline as clay_timeline
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui import viewport as clay_viewport

RECT = (0.0, 0.0, 320.0, 240.0)


class _State:
    def __init__(self) -> None:
        self.tool = "select"
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": _State()})()

    def toast(self, *_a: object, **_k: object) -> None:
        pass


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


def test_active_gizmo_is_none_when_every_element_pick_is_on_a_hidden_object(view) -> None:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.select([uid])
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=[0]))
    view.frame_whole(doc)
    view.draw(doc, RECT, 0.0)
    assert view.active_gizmo(doc) is not None  # visible: the handle is on offer

    doc.set_props(uid, visible=False)  # the outliner's eye, to see inside it
    view.draw(doc, RECT, 0.0)

    assert view.active_gizmo(doc) is None, "a hidden object's element gizmo still hit-tests"


def test_playback_does_not_advance_the_playhead_under_a_live_pose_drag(view) -> None:
    doc = bd.ClayDoc()
    hip = doc.add_bone().uid
    doc.add_animation("Clip")
    doc.set_key(hip, "translation", 0.0, [0, 0, 0])
    doc.set_key(hip, "translation", 2.0, [0, 0, 0])
    doc.set_pose_mode(True)
    doc.select([hip])
    doc.auto_key = False
    doc.set_rig_time(0.5)
    assert clay_timeline.toggle_play(doc)
    view.frame_whole(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (160.0, 120.0)
    assert view.begin_keyboard_drag(doc, "move")
    view._drag_keyboard(doc, (200.0, 100.0))
    held = {k: sorted(p) for k, p in doc.live_pose.items()}
    assert held, "the drag holds a live pose"

    moved = clay_viewport.advance_playback(view, doc, 1 / 60)

    assert not moved
    assert doc.rig_time == pytest.approx(0.5), "the playhead moved under a held pose"
    assert {k: sorted(p) for k, p in doc.live_pose.items()} == held, "the tick wiped the held pose"
    view.cancel_drag(doc)
    # And once the drag is over the clip plays on.
    assert clay_viewport.advance_playback(view, doc, 1 / 60)
    assert np.isclose(doc.rig_time, 0.5 + 1 / 60)
