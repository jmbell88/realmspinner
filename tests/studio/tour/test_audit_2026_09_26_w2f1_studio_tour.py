"""Regression test for the 2026-09-26 audit, finding tour-1-01.

The Inker Basics tour's "toolbox" step told the reader "B is the brush, and B
again is the spray" -- promising a letter-cycling gesture no binding
implements. Chapter 5 of the manual carried the identical claim until the
2026-09-05 audit's finding docs-02 fixed it (pinned by
``tests/manual/test_manual_promises.py::test_chapter05_tool_group_prose_matches_actual_key_bindings``);
this tour step was never updated to match.
"""

from __future__ import annotations

from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.tour import TOURS


def _toolbox_step():
    for tour in TOURS:
        for step in tour.steps:
            if step.id == "toolbox":
                return step
    raise AssertionError("the Inker Basics tour lost its 'toolbox' step")


def test_inker_tour_toolbox_step_does_not_promise_letter_cycling():
    step = _toolbox_step()
    assert "again cycles" not in step.body, (
        "the toolbox step still describes pressing a group's letter again to "
        "cycle within it; no binding implements that"
    )
    assert "B is the brush, and B" not in step.body, (
        "the toolbox step still tells the reader B twice gives Spray"
    )

    # Sanity, ``test_chapter05_tool_group_prose_matches_actual_key_bindings``'s
    # own check: spray never answers to a bare second B.
    chords: dict[str, list[str]] = {}
    for binding in inker_ops._TOOL_BINDINGS:
        chords.setdefault(binding.target, []).append(binding.chord)
    assert "spray" in chords
    assert "B" not in chords["spray"]
