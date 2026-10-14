"""The 2026-10-14 audit, finding shell-chrome-07: the palette's Undo and Redo
rows stayed lit while Clay refused them (a save in flight, a live gesture), and
the press returned silently. They now grey with the mode's own sentence."""

from __future__ import annotations

import pytest
from modes.clay.test_clay_palette import _ctx

from realmspinner.studio import palette
from realmspinner.studio.modes.clay import doc_io


def _rows(ctx):
    return {c.key: c for c in palette.commands(ctx) if c.key in ("undo", "redo")}


def _with_history(ctx):
    """A document that can both undo and redo, so the only thing that can grey
    the rows is the mode's refusal."""
    doc = ctx.state.clay.active.doc
    doc.select([doc.objects[1].uid])
    doc.undo()
    assert doc.history.can_undo and doc.history.can_redo
    return ctx


@pytest.mark.parametrize("key", ["undo", "redo"])
def test_the_palette_undo_and_redo_rows_stay_lit_when_clay_is_idle(key):
    ctx = _with_history(_ctx())
    row = _rows(ctx)[key]
    assert row.enabled(ctx) is True


@pytest.mark.parametrize("key", ["undo", "redo"])
def test_the_palette_undo_and_redo_rows_grey_with_the_reason_while_clay_saves(key):
    ctx = _with_history(_ctx())
    ctx.state.clay.active.saving = True
    row = _rows(ctx)[key]
    assert row.enabled(ctx) is False
    assert row.why == "Saving..."


@pytest.mark.parametrize("key", ["undo", "redo"])
def test_the_palette_undo_and_redo_rows_grey_with_the_reason_during_a_live_gesture(key):
    ctx = _with_history(_ctx(with_view=True))
    ctx.clay_view.dragging = True
    row = _rows(ctx)[key]
    assert row.enabled(ctx) is False
    assert row.why == doc_io.GESTURE_WHY
