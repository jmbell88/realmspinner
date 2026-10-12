"""The 2026-10-10 audit's clay-25, drag half: a drag preview is drawn once a frame.

The hover picks were throttled and the two drag previews were not -- the element
preview (an affine over every moved vertex plus a buffer write) and the op-drag
update (a kernel run) ran once per MOUSEMOTION event, ~16 a frame from a high-rate
mouse, and all but the last were overwritten unseen. A distance throttle would have
left the committed value up to 3 px behind the pointer, so the work is coalesced to
one per frame instead and drawn before the commit, so the value that lands is exact.

Rides the session ``gl`` fixture like its siblings, and skips where there is no GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 128.0, 96.0)


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None

    def toast(self, message: str, level: str = "info") -> None:
        pass


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def view(gl):
    ctx = _Ctx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


def _faces_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    return doc, uid


def _ready(view: Any, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)


def _count(view: Any, monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    calls: list[int] = []
    real = getattr(view, name)
    monkeypatch.setattr(view, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_an_element_drag_draws_one_preview_a_frame_however_many_motion_events_arrive(
    view, monkeypatch
) -> None:
    doc, _uid = _faces_doc()
    _ready(view, doc)
    assert view.begin_keyboard_drag(doc, "move")
    previews = _count(view, monkeypatch, "_preview_element_drag")

    for x in range(70, 100, 2):  # fifteen events between two frames
        view._motion(doc, (float(x), 48.0))
    assert previews == [], "an event drew a preview of its own"

    view.draw(doc, RECT, 0.0)
    assert len(previews) == 1, "the frame did not draw exactly one preview"
    view.draw(doc, RECT, 0.0)
    assert len(previews) == 1, "a frame with no new motion drew the old one again"


def test_an_op_drag_runs_its_update_once_a_frame_however_many_motion_events_arrive(
    view, monkeypatch
) -> None:
    doc, _uid = _faces_doc()
    _ready(view, doc)
    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    updates = _count(view, monkeypatch, "_op_drag_motion")

    for x in range(70, 100, 2):
        view._motion(doc, (float(x), 48.0))
    assert updates == [], "an event ran an update of its own"

    view.draw(doc, RECT, 0.0)
    assert len(updates) == 1


@pytest.mark.parametrize("how", ["release", "settle"])
def test_a_release_or_a_settle_commits_the_newest_pointer_position_with_no_frame_between(
    view, how: str
) -> None:
    """The reference moves with a frame after every event; the subject queues every
    event and lets the commit draw the last. Both must land on the same positions."""
    reference, ref_uid = _faces_doc()
    _ready(view, reference)
    assert view.begin_keyboard_drag(reference, "move")
    for x in range(70, 100, 2):
        view._motion(reference, (float(x), 48.0))
        view.flush_drag_preview(reference)
    view.settle_drag(reference)

    subject, uid = _faces_doc()
    _ready(view, subject)
    assert view.begin_keyboard_drag(subject, "move")
    for x in range(70, 100, 2):
        view._motion(subject, (float(x), 48.0))
    if how == "release":
        assert view._release_drag(subject, 1)
    else:
        assert view.settle_drag(subject)

    moved = np.array(subject.by_uid(uid).mesh.positions)
    assert not np.array_equal(moved, np.array(bp.box().positions)), "the drag moved nothing"
    assert np.array_equal(moved, np.array(reference.by_uid(ref_uid).mesh.positions))


def test_cancelling_a_drag_drops_the_queued_preview_instead_of_drawing_it_next_frame(
    view, monkeypatch
) -> None:
    doc, uid = _faces_doc()
    _ready(view, doc)
    before = np.array(doc.by_uid(uid).mesh.positions)
    assert view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (95.0, 48.0))
    previews = _count(view, monkeypatch, "_preview_element_drag")

    assert view.cancel_drag(doc)
    view.draw(doc, RECT, 0.0)

    assert previews == [], "the cancelled drag's last position was drawn after Esc"
    assert np.array_equal(np.array(doc.by_uid(uid).mesh.positions), before)
