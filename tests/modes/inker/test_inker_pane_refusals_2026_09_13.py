"""Two pane doors the 2026-09-13 audit found swallowing a refusal.

``inker_ops.run`` (finding inker-01, closed in 9a691183) is the one choke
point every menu row, shortcut and gesture funnels through, and it catches a
document method that refuses by *raising* ``ValueError``. But the timeline's
row menu and the palette pane's Remove button call the document directly --
they do not go through ``inker_ops.run`` -- so each needed its own catch.
Finding inker-06 is that neither had one: a per-cel Z lift makes
``merge_range`` raise, and the transparent palette slot makes ``remove_slot``
raise, and both refusals unwound silently through the pane guard.

The row menu and the Remove button both live behind real imgui popups this
suite does not open, so what is regression-tested here is the extracted door
each draw call now goes through -- ``inker_timeline.merge_range_or_say`` and
``inker_colors.remove_slot_or_say``/``remove_slot_gate`` -- which is exactly
the code the click runs. Neither function existed before this fix: importing
them from the unfixed tree fails with ``AttributeError``, which is itself the
first form this regression took.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from realmspinner.kernels import pixel as inker
from realmspinner.studio.modes.inker.ui.panes import colors as inker_colors
from realmspinner.studio.modes.inker.ui.panes import timeline as inker_timeline


class _Sayer:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


def _ctx():
    state = _Sayer()
    return SimpleNamespace(state=SimpleNamespace(inker=state)), state


# --- inker-06a: merge down across a per-cel Z lift --------------------------


def _lifted_anim_doc():
    """Two tracks whose frame-0 cels are inverted by a cel-z lift, so
    ``merge_range(1, 2)`` -- the row menu's own call, ``min(rows), max(rows)``
    -- reaches ``_refuse_merge_across_z`` and raises."""
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    doc.add_layer("L1")
    doc.add_layer("L2")
    doc.ensure_animation()
    doc.add_frame()
    assert doc.set_cel_z(-5, track_index=2, frame_index=0)
    return doc


def test_merge_down_on_a_lifted_cel_says_why_rather_than_doing_nothing():
    doc = _lifted_anim_doc()
    ctx, sayer = _ctx()

    result = inker_timeline.merge_range_or_say(ctx, doc, 1, 2)

    assert result is False
    assert len(sayer.said) == 1
    assert "lifts these cels apart" in sayer.said[0]
    # And the stack really is untouched -- the refusal did nothing, silently
    # doing nothing is exactly the defect, but the merge itself must also not
    # have half-applied.
    assert len(doc.stack) == 3


def test_merge_down_still_merges_when_nothing_refuses():
    """The catch must not turn a normal merge into a no-op."""
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    doc.add_layer("L1")
    doc.add_layer("L2")
    ctx, sayer = _ctx()

    result = inker_timeline.merge_range_or_say(ctx, doc, 1, 2)

    assert result is True
    assert sayer.said == []
    assert len(doc.stack) == 2


# --- inker-06b: Remove on the transparent palette slot ----------------------


def _indexed_doc_with_hole():
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    doc.convert_to_indexed(
        [(0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)], transparent=0
    )
    assert doc.is_indexed
    assert doc.transparent_index == 0
    return doc


def test_removing_the_transparent_palette_slot_is_greyed_with_a_reason():
    doc = _indexed_doc_with_hole()
    hole = doc.transparent_index

    removable, reason = inker_colors.remove_slot_gate(len(doc.palette), hole, hole)

    assert removable is False
    # Before the fix, every refusal cited "keeps at least one colour" -- untrue
    # here, since this palette has three. The reason must name the real cause.
    assert "keeps at least one colour" not in reason
    assert "transparent" in reason.lower()


def test_removing_the_only_slot_of_a_palette_with_no_hole_cites_the_count():
    """The other branch of the same gate: a palette-constrained document with
    no transparent index (``hole == -1``) still refuses its last slot, and
    that refusal's reason is the true one."""
    removable, reason = inker_colors.remove_slot_gate(1, 0, -1)

    assert removable is False
    assert "keeps at least one colour" in reason


def test_removing_an_ordinary_slot_with_colours_to_spare_is_not_greyed():
    doc = _indexed_doc_with_hole()
    other = next(i for i in range(len(doc.palette)) if i != doc.transparent_index)

    removable, _reason = inker_colors.remove_slot_gate(
        len(doc.palette), other, doc.transparent_index
    )

    assert removable is True


def test_remove_slot_or_say_reports_the_transparent_refusal_instead_of_doing_nothing():
    doc = _indexed_doc_with_hole()
    hole = doc.transparent_index
    _ctx_obj, sayer = _ctx()
    state = SimpleNamespace(say=sayer.say)

    result = inker_colors.remove_slot_or_say(state, doc, hole)

    assert result is False
    assert len(sayer.said) == 1
    assert "transparent" in sayer.said[0].lower()
    # Untouched: the palette still holds the hole.
    assert doc.transparent_index == hole


def test_remove_slot_or_say_still_removes_an_ordinary_slot():
    doc = _indexed_doc_with_hole()
    before = len(doc.palette)
    _ctx_obj, sayer = _ctx()
    state = SimpleNamespace(say=sayer.say)
    other = next(i for i in range(len(doc.palette)) if i != doc.transparent_index)

    result = inker_colors.remove_slot_or_say(state, doc, other)

    assert result is True
    assert sayer.said == []
    assert len(doc.palette) == before - 1
