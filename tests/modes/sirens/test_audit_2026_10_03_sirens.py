"""Audit 2026-10-03 sirens-01..02 regressions."""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import synth


def _four_patterns():
    doc = D.new_song()
    first = doc.patterns[0]
    uids = [first.uid] + [doc.add_pattern(rows=8).uid for _ in range(3)]
    doc.set_order(uids)
    return doc, uids


def test_deleting_an_order_entry_above_the_loop_point_keeps_the_loop_on_the_same_entry():
    doc, uids = _four_patterns()
    doc.set_song(loop_order=2)  # looping from C
    doc.set_order([uids[1], uids[2], uids[3]])  # A deleted
    assert doc.order[doc.loop_order] == uids[2]
    doc.undo() if hasattr(doc, "undo") else doc.history.undo(doc)
    assert doc.loop_order == 2


def test_deleting_a_pattern_above_the_loop_point_keeps_the_loop_on_the_same_entry():
    doc, uids = _four_patterns()
    doc.set_song(loop_order=2)
    doc.remove_pattern(uids[0])
    assert doc.order[doc.loop_order] == uids[2]


def test_a_song_past_the_render_ceiling_is_refused_not_truncated(monkeypatch):
    monkeypatch.setattr(synth, "MAX_RENDER_SECONDS", 0.5)
    doc = D.new_song()
    doc.resize_pattern(doc.patterns[0].uid, 64)
    doc.set_order([doc.patterns[0].uid] * 64)
    with pytest.raises(ValueError, match="render limit"):
        synth.render(doc)
