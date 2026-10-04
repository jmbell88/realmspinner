"""The 2026-10-03 audit's Medium tour findings (tour-02, tour-03, tour-04).

Each test's name is the claim. All three are the same family of defect: a step
card that says something the tour machinery does not do, so a reader following
the card to the letter gets a different result from the one it promised.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker.state import InkerState
from realmspinner.studio.panes import tour as tour_pane
from realmspinner.studio.tour import TOURS

SRC = Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "studio"


def _steps():
    for tour in TOURS:
        for step in tour.steps:
            yield tour, step


def _fresh_session_at(step):
    """What ``satisfied`` sees the moment the reader *reaches* ``step``.

    A step that names a mode has put the reader in it (the tour switches to it
    on arrival); one that names none is reached from wherever the previous
    step left them, and the earliest that can be is Home. No document is open,
    no setting has been touched -- a fresh ``InkerState`` is exactly what a
    first-time reader's session holds.
    """
    return SimpleNamespace(
        state=SimpleNamespace(
            mode=step.mode or "home",
            inker=InkerState(),
            sirens=None,
        )
    )


def test_no_tour_step_is_already_satisfied_by_a_fresh_session_at_the_moment_it_is_reached():
    """The 2026-10-03 audit, finding tour-02: ``inker-basics``' "Take the
    brush" waited on ``tool_is brush``, and a fresh ``InkerState`` already
    holds the brush, so the card read "Done." on its first frame -- before the
    reader pressed B or clicked anything. The 2026-09-16 guard only forbade
    ``mode_is`` naming the step's own mode; this is the general form of it.
    """
    for tour, step in _steps():
        if step.done.name == "manual":
            continue
        # A step whose whole job is to carry the reader into its own mode is
        # reached *from the rail*, a mode earlier; its ``mode`` is None, so
        # ``_fresh_session_at`` already starts it on Home.
        ctx = _fresh_session_at(step)
        assert not tour_pane.satisfied(ctx, step.done.name, step.done.arg), (
            f"{tour.key}/{step.id}: done={step.done!r} already holds in a fresh "
            "session, so the card reads 'Done.' the instant it appears"
        )


def test_no_step_copy_promises_the_tour_will_advance_on_its_own():
    """The 2026-10-03 audit, finding tour-03: ``open-create`` said "and I will
    carry on", but nothing advances a tour when its condition is met -- the
    card only switches "Waiting for you." to "Done." and Next stays enabled
    throughout. A reader who clicked Create as told sat on a card that did not
    move. The copy may say the card will read Done; it may not say the tour
    moves.
    """
    promise = re.compile(
        r"\b(?:I|we|the tour|this tour|the card)(?:\s+will|'ll)\s+"
        r"(?:carry on|continue|move on|go on|advance|proceed|take you)",
        re.IGNORECASE,
    )
    for tour, step in _steps():
        found = promise.search(step.body)
        assert found is None, (
            f"{tour.key}/{step.id}: copy promises the tour advances by itself "
            f"({found.group(0)!r}); only the card's Next does"
        )


def test_a_point_and_wait_step_tells_the_reader_that_next_is_what_moves_on():
    """tour-03, the other half: ``open-create`` is the one point-and-wait step
    in ``first-hour``'s opening, and the reader needs to be told the card ends
    at "Done." and Next is theirs to press."""
    tour = next(t for t in TOURS if t.key == "first-hour")
    step = next(s for s in tour.steps if s.id == "open-create")
    assert "Next" in step.body, step.body


def test_the_animate_step_points_at_the_surface_that_holds_animate_this_drawing():
    """The 2026-10-03 audit, finding tour-04: the step said "Press Animate this
    drawing" and rang the timeline panel, but the op is a Frame-menu row and no
    button of that name is drawn in the timeline (which is not even drawn on a
    still document). The copy has to name the menu the op lives in, and a ring
    may sit only on a pane whose source actually draws the named control.
    """
    menu = inker_ops.get("animate").menu
    tour = next(t for t in TOURS if t.key == "inker-basics")
    step = next(s for s in tour.steps if s.id == "animate")
    assert f"{menu} menu" in step.body, (
        f"the body does not say where the control is ({menu!r} menu): {step.body!r}"
    )
    if step.anchor is not None:
        pane_source = (
            SRC / "modes" / "inker" / "ui" / "panes" / "timeline.py"
        ).read_text(encoding="utf-8")
        assert "Animate this drawing" in pane_source, (
            f"the step rings {step.anchor!r}, whose pane draws no Animate control"
        )

