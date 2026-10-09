"""Clay's input model: a plain drag orbits in every mode.

The rule (``studio/modes/clay/gestures.py``): a plain drag orbits, a click
without a drag picks or clears, Shift+drag sweeps a box that adds, Ctrl+drag one
that subtracts, and Alt+drag orbits whatever is under the pointer. Before it,
Points/Edges/Faces with the Select tool swept a box on a plain drag (so a first
look-around drew a rectangle and cleared the picks) and object mode cleared the
selection at the *press* (so the first orbit drag deselected the object).

The classifier is pure and walked as a table; the view half rides the session
``gl`` fixture like every sibling and skips where there is no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import gestures
from realmspinner.studio.modes.clay.gestures import Press
from realmspinner.studio.modes.clay.ui import view as clay_view

# --- the classifier (pure) -----------------------------------------------------

_ORBIT, _ADD, _SUB = gestures.ORBIT, gestures.MARQUEE_ADD, gestures.MARQUEE_SUB
_PICK, _CLEAR = gestures.PICK, gestures.CLEAR

#: (element_mode, under, tool, shift, ctrl) -> Press(now, drag, click)
TABLE = [
    # object mode: a click on an object picks at once; a plain drag on nothing
    # orbits and only a click clears; Shift/Ctrl on nothing neither sweep nor clear.
    (False, "object", "select", False, False, Press(_PICK, _ORBIT, "")),
    (False, "object", "move", True, False, Press(_PICK, _ORBIT, "")),
    (False, "object", "select", False, True, Press(_PICK, _ORBIT, "")),
    (False, "nothing", "select", False, False, Press("", _ORBIT, _CLEAR)),
    (False, "nothing", "move", False, False, Press("", _ORBIT, _CLEAR)),
    (False, "nothing", "select", True, False, Press("", _ORBIT, "")),
    (False, "nothing", "select", False, True, Press("", _ORBIT, "")),
    # element modes, empty space: plain orbits (and a click clears under Select
    # only), Shift sweeps to add, Ctrl sweeps to subtract, under every tool.
    (True, "nothing", "select", False, False, Press("", _ORBIT, _CLEAR)),
    (True, "nothing", "move", False, False, Press("", _ORBIT, "")),
    (True, "nothing", "rotate", False, False, Press("", _ORBIT, "")),
    (True, "nothing", "scale", False, False, Press("", _ORBIT, "")),
    (True, "nothing", "select", True, False, Press("", _ADD, "")),
    (True, "nothing", "move", True, False, Press("", _ADD, "")),
    (True, "nothing", "select", False, True, Press("", _SUB, "")),
    (True, "nothing", "scale", False, True, Press("", _SUB, "")),
    (True, "nothing", "select", True, True, Press("", _ADD, "")),
    # element modes, a surface with no element in reach: a click clears that
    # object's picks under every tool; a drag orbits, or sweeps with a modifier.
    (True, "surface", "select", False, False, Press("", _ORBIT, _CLEAR)),
    (True, "surface", "move", False, False, Press("", _ORBIT, _CLEAR)),
    (True, "surface", "select", True, False, Press("", _ADD, "")),
    (True, "surface", "select", False, True, Press("", _SUB, "")),
    # element modes, on an element: a plain press picks at once; Shift/Ctrl wait
    # for the release so a box can start on the model.
    (True, "element", "select", False, False, Press(_PICK, _ORBIT, "")),
    (True, "element", "move", False, False, Press(_PICK, _ORBIT, "")),
    (True, "element", "select", True, False, Press("", _ADD, _PICK)),
    (True, "element", "select", False, True, Press("", _SUB, _PICK)),
]


@pytest.mark.parametrize(("element_mode", "under", "tool", "shift", "ctrl", "expected"), TABLE)
def test_the_classifier_table(element_mode, under, tool, shift, ctrl, expected) -> None:
    got = gestures.classify_press(
        element_mode=element_mode, under=under, tool=tool, shift=shift, ctrl=ctrl
    )
    assert got == expected


def test_a_plain_press_never_starts_a_marquee_anywhere() -> None:
    for element_mode in (False, True):
        for under in gestures.UNDER:
            if (under == "object") == element_mode:
                continue  # an object is only under the pointer in object mode, and vice versa
            for tool in ("select", "move", "rotate", "scale"):
                got = gestures.classify_press(
                    element_mode=element_mode, under=under, tool=tool, shift=False, ctrl=False
                )
                assert got.drag == gestures.ORBIT, (element_mode, under, tool)


def test_object_mode_has_no_box_select_whatever_is_held() -> None:
    for under in ("object", "nothing"):
        for shift, ctrl in ((True, False), (False, True), (True, True)):
            got = gestures.classify_press(
                element_mode=False, under=under, tool="select", shift=shift, ctrl=ctrl
            )
            assert got.drag == gestures.ORBIT


def test_an_unknown_target_is_refused() -> None:
    with pytest.raises(ValueError):
        gestures.classify_press(
            element_mode=True, under="sky", tool="select", shift=False, ctrl=False
        )


def test_the_click_slop_is_four_pixels_and_is_a_strict_bound() -> None:
    assert gestures.CLICK_SLOP == 4.0
    assert gestures.is_click(0.0) and gestures.is_click(3.99)
    assert not gestures.is_click(4.0) and not gestures.is_click(40.0)


def test_pick_how_gives_shift_the_tie() -> None:
    assert gestures.pick_how(False, False) == "replace"
    assert gestures.pick_how(True, False) == "add"
    assert gestures.pick_how(False, True) == "subtract"
    assert gestures.pick_how(True, True) == "add"


# --- the viewport ----------------------------------------------------------------

RECT = (0.0, 0.0, 200.0, 150.0)
CORNER = (2, 2)
FAR = (int(RECT[2]) - 1, int(RECT[3]) - 1)
CENTRE = (int(RECT[2] * 0.5), int(RECT[3] * 0.5))


class _State:
    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _Ctx:
    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()


@pytest.fixture(autouse=True)
def _mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    return monkeypatch


def _hold(monkeypatch, mods: int) -> None:
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods, raising=False)


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


def _doc(mode: str = "object") -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="box", mesh=bp.box(), translation=m3.vec3()))
    doc.set_element_mode(mode)
    return doc


def _ready(view, doc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)


def _send(view, doc, kind: str, pos, button: int = 1) -> None:
    import pygame

    kinds = {
        "down": pygame.MOUSEBUTTONDOWN,
        "up": pygame.MOUSEBUTTONUP,
        "move": pygame.MOUSEMOTION,
    }
    fields = {"pos": pos} if kind == "move" else {"button": button, "pos": pos}
    view.handle_event(doc, pygame.event.Event(kinds[kind], **fields), True)


def _drag(view, doc, start, end, steps: int = 4) -> None:
    """Press at *start*, move to *end* in a few steps, release."""
    _send(view, doc, "down", start)
    for i in range(1, steps + 1):
        t = i / steps
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        _send(view, doc, "move", (x, y))
    _send(view, doc, "up", end)


def _eye(view) -> np.ndarray:
    """Where the camera is *going*: an orbit sets the goal angles and the
    position eases toward them over the following frames."""
    return np.array([view.camera._goal_theta, view.camera._goal_phi], dtype="f8")


# -- object mode -----------------------------------------------------------------


def test_object_mode_a_plain_drag_on_empty_space_orbits_and_does_not_deselect(view) -> None:
    """Fails on the old code: the press on empty space ran ``doc.select([])``
    straight away, so ``doc.selection`` was empty by the first motion."""
    doc = _doc()
    uid = doc.objects[0].uid
    doc.select([uid])
    _ready(view, doc)
    eye = _eye(view)

    _send(view, doc, "down", CORNER)
    assert doc.selection == {uid}, "the press alone must not decide"
    _send(view, doc, "move", (CORNER[0] + 40, CORNER[1] + 20))
    _send(view, doc, "up", (CORNER[0] + 40, CORNER[1] + 20))

    assert doc.selection == {uid}
    assert not np.allclose(eye, _eye(view)), "the drag did not orbit"
    assert view.marquee is None


def test_object_mode_a_click_without_a_drag_on_empty_space_still_deselects(view) -> None:
    doc = _doc()
    doc.select([doc.objects[0].uid])
    _ready(view, doc)

    _send(view, doc, "down", CORNER)
    _send(view, doc, "up", CORNER)

    assert doc.selection == set()


def test_object_mode_a_wobble_inside_the_slop_is_still_a_click(view) -> None:
    doc = _doc()
    doc.select([doc.objects[0].uid])
    _ready(view, doc)

    _drag(view, doc, CORNER, (CORNER[0] + 2, CORNER[1] + 1), steps=2)

    assert doc.selection == set()


def test_object_mode_a_drag_that_returns_to_the_press_point_is_still_a_drag(view) -> None:
    doc = _doc()
    uid = doc.objects[0].uid
    doc.select([uid])
    _ready(view, doc)

    _send(view, doc, "down", CORNER)
    _send(view, doc, "move", (CORNER[0] + 30, CORNER[1]))
    _send(view, doc, "move", CORNER)
    _send(view, doc, "up", CORNER)

    assert doc.selection == {uid}


def test_object_mode_shift_drag_on_empty_space_orbits_and_draws_no_box(view, _mods) -> None:
    import pygame

    doc = _doc()
    uid = doc.objects[0].uid
    doc.select([uid])
    _ready(view, doc)
    eye = _eye(view)

    _hold(_mods, pygame.KMOD_SHIFT)
    _send(view, doc, "down", CORNER)
    _send(view, doc, "move", (CORNER[0] + 40, CORNER[1] + 20))
    assert view._grab == "orbit" and view.marquee is None
    _send(view, doc, "up", (CORNER[0] + 40, CORNER[1] + 20))

    assert doc.selection == {uid}
    assert not np.allclose(eye, _eye(view))


def test_object_mode_a_click_on_an_object_still_selects_it(view) -> None:
    doc = _doc()
    _ready(view, doc)

    _send(view, doc, "down", CENTRE)
    _send(view, doc, "up", CENTRE)

    assert doc.selection == {doc.objects[0].uid}


def test_alt_drag_orbits_in_object_mode_and_selects_nothing(view, _mods) -> None:
    import pygame

    doc = _doc()
    _ready(view, doc)
    eye = _eye(view)

    _hold(_mods, pygame.KMOD_ALT)
    _drag(view, doc, CENTRE, (CENTRE[0] + 40, CENTRE[1] + 20))

    assert doc.selection == set()
    assert not np.allclose(eye, _eye(view))


# -- element modes ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["vertex", "edge", "face"])
def test_element_modes_a_plain_drag_on_empty_space_orbits_and_leaves_the_picks(view, mode) -> None:
    """Fails on the old code: under the Select tool the press started a marquee
    (``_grab == "marquee"``) and the release's zero-area clear wiped the picks."""
    doc = _doc(mode)
    uid = doc.objects[0].uid
    one = {
        "vertex": el.ElementSel(verts=[0]),
        "edge": el.ElementSel(edges=[[0, 1]]),
        "face": el.ElementSel(faces=[0]),
    }[mode]
    doc.set_element_sel(uid, one)
    _ready(view, doc)
    picks = doc.element_sel_of(uid)
    eye = _eye(view)

    _send(view, doc, "down", CORNER)
    assert view._grab == "orbit"
    _send(view, doc, "move", (CORNER[0] + 40, CORNER[1] + 20))
    _send(view, doc, "up", (CORNER[0] + 40, CORNER[1] + 20))

    assert view.marquee is None
    assert doc.element_sel_of(uid).count(mode) == picks.count(mode) == 1
    assert not np.allclose(eye, _eye(view))


@pytest.mark.parametrize("tool", ["move", "rotate", "scale"])
def test_the_transform_tools_still_orbit_on_a_plain_drag_and_keep_the_picks(gl, tool) -> None:
    v = clay_view.ClayView(gl, _Ctx(tool))
    try:
        doc = _doc("face")
        uid = doc.objects[0].uid
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
        _ready(v, doc)
        eye = _eye(v)

        _drag(v, doc, CORNER, (CORNER[0] + 40, CORNER[1] + 20))

        assert doc.element_sel_of(uid).count("face") == 1, "a drag on empty space keeps picks"
        assert not np.allclose(eye, _eye(v))
    finally:
        v.release()


def test_element_modes_a_click_on_empty_space_clears_the_picks_under_select(view) -> None:
    doc = _doc("face")
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    _send(view, doc, "down", CORNER)
    assert doc.element_sel, "the press alone must not decide"
    _send(view, doc, "up", CORNER)

    assert doc.element_sel == {}


def test_element_modes_a_click_on_an_element_still_picks_it(view) -> None:
    doc = _doc("face")
    _ready(view, doc)

    _send(view, doc, "down", CENTRE)
    _send(view, doc, "up", CENTRE)

    assert len(doc.element_sel_of(doc.objects[0].uid).faces) == 1


def test_element_modes_shift_drag_sweeps_a_box_that_adds_to_the_picks(view, _mods) -> None:
    import pygame

    doc = _doc("face")
    uid = doc.objects[0].uid
    _ready(view, doc)
    _send(view, doc, "down", CENTRE)
    _send(view, doc, "up", CENTRE)
    kept = doc.element_sel_of(uid).faces.tolist()
    assert len(kept) == 1

    _hold(_mods, pygame.KMOD_SHIFT)
    _send(view, doc, "down", (1, 1))
    assert view._grab == "marquee"
    _send(view, doc, "move", FAR)
    assert view.marquee is not None
    _send(view, doc, "up", FAR)

    assert view.marquee is None
    assert len(doc.element_sel_of(uid).faces) == 6
    assert set(kept) <= set(doc.element_sel_of(uid).faces.tolist())


def test_element_modes_ctrl_drag_sweeps_a_box_that_subtracts_from_the_picks(view, _mods) -> None:
    import pygame

    doc = _doc("face")
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0, 1, 2, 3, 4, 5]))
    _ready(view, doc)

    _hold(_mods, pygame.KMOD_CTRL)
    _send(view, doc, "down", (1, 1))
    assert view._grab == "marquee"
    _send(view, doc, "move", FAR)
    _send(view, doc, "up", FAR)

    assert doc.element_sel_of(uid).count("face") == 0


@pytest.mark.parametrize("held", ["shift", "ctrl"])
def test_a_zero_area_modified_marquee_leaves_the_picks_alone(view, _mods, held) -> None:
    import pygame

    doc = _doc("face")
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    _ready(view, doc)

    _hold(_mods, pygame.KMOD_SHIFT if held == "shift" else pygame.KMOD_CTRL)
    _send(view, doc, "down", CORNER)
    _send(view, doc, "up", CORNER)

    assert doc.element_sel_of(uid).count("face") == 1


def test_a_replace_marquee_with_no_area_no_longer_clears_the_selection(view) -> None:
    """Fails on the old code, where a zero-area *replace* marquee was how a
    click on empty space deselected."""
    doc = _doc("face")
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    _ready(view, doc)
    view._grab = "marquee"
    view._marquee_from = (50.0, 50.0)
    view.marquee = (50.0, 50.0, 50.0, 50.0)
    view._marquee_add = "replace"

    view._release_drag(doc, 1)

    assert doc.element_sel_of(uid).count("face") == 1


def test_a_shift_drag_may_begin_on_the_model_itself(view, _mods) -> None:
    """In Faces mode the model is nearly all there is to start on: a Shift press
    on a face waits for the release, so it can be the corner of a box."""
    import pygame

    doc = _doc("face")
    _ready(view, doc)

    _hold(_mods, pygame.KMOD_SHIFT)
    _send(view, doc, "down", CENTRE)
    assert view._grab == "marquee"
    assert not doc.element_sel, "a Shift press on an element must not pick before the release"
    _send(view, doc, "move", FAR)
    assert view.marquee == (float(CENTRE[0]), float(CENTRE[1]), float(FAR[0]), float(FAR[1]))
    _send(view, doc, "up", FAR)

    assert view.marquee is None and view._grab is None
    assert view._click is None


def test_alt_drag_orbits_in_an_element_mode_and_picks_nothing(view, _mods) -> None:
    import pygame

    doc = _doc("face")
    _ready(view, doc)
    eye = _eye(view)

    _hold(_mods, pygame.KMOD_ALT)
    _drag(view, doc, CENTRE, (CENTRE[0] + 40, CENTRE[1] + 20))

    assert doc.element_sel == {}
    assert not np.allclose(eye, _eye(view))


def test_a_press_on_a_surface_with_no_element_in_reach_waits_to_clear(view) -> None:
    """Vertex mode: the middle of a face is a surface, with no vertex near it.
    The clear that used to run at the press now runs on a click's release."""
    doc = _doc("vertex")
    uid = doc.objects[0].uid
    _ready(view, doc)
    spot = next(
        (x, y)
        for y in range(10, int(RECT[3]) - 10, 3)
        for x in range(10, int(RECT[2]) - 10, 3)
        if (hit := view.pick_face(doc, (x, y))) is not None
        and view.pick_element(doc, (x, y), hit) is None
    )

    _send(view, doc, "down", spot)
    assert view._click == ("clear-surface", uid)
    assert view._grab == "orbit"
    _send(view, doc, "move", (spot[0] + 30, spot[1]))
    _send(view, doc, "up", (spot[0] + 30, spot[1]))
    assert view._click is None


def test_a_click_that_arrives_after_the_mode_left_its_kind_is_dropped(view) -> None:
    doc = _doc("face")
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    _ready(view, doc)
    _send(view, doc, "down", CORNER)
    doc.set_element_mode("object")
    doc.select([uid])

    _send(view, doc, "up", CORNER)

    assert doc.selection == {uid}, "an element clear must not deselect objects"
    assert doc.element_sel == {}
