"""the 2026-09-26 audit, finding inker-mode-19: ``can_propagate`` and
``propagate_reason`` each ran a full-cel ``mirror.changed_weight`` diff, so
checking one disabled row -- a control's ``enabled``, then its greyed
``reason``, exactly what ``ui/panes/sheet.py``'s ``_press`` does on every
draw -- diffed the whole cel twice. This asserts the diff runs once per
``sync_mark`` tick no matter how many of the two are asked, and that a real
edit (a new tick) still gets a fresh answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from realmspinner.kernels.pixel import mirror
from realmspinner.kernels.pixel.sheetin import document_from_sheet
from realmspinner.studio.modes.inker import sheet as inker_sheet
from realmspinner.studio.modes.inker import state as inker_state

CELL = 16
DIRECTIONS = ("front", "left", "back", "right")


def _sheet_doc():
    count = len(DIRECTIONS) * 2
    atlas = np.zeros((CELL, count * CELL, 4), dtype=np.uint8)
    cells, tags = [], []
    for d, direction in enumerate(DIRECTIONS):
        tags.append({"name": f"walk_{direction}", "start": d * 2, "end": d * 2 + 1, "loop": True})
        for f in range(2):
            index = d * 2 + f
            atlas[4:12, index * CELL + 6 : index * CELL + 10] = (40, 60, 200, 255)
            cells.append({"x": index * CELL, "y": 0, "w": CELL, "h": CELL})
    return document_from_sheet(atlas, cells, {"tags": tags, "frames": []})


def _scene():
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=_sheet_doc(), uid="t")
    state.docs.append(tab)
    state.active_uid = "t"
    return state, tab


def _counting_diff(monkeypatch, calls: list[int]):
    real = mirror.changed_weight

    def counting(before, now):
        calls.append(1)
        return real(before, now)

    monkeypatch.setattr(inker_sheet.mirror, "changed_weight", counting)


def test_one_disabled_row_check_diffs_the_cel_once(monkeypatch):
    state, tab = _scene()
    inker_sheet.sync_mark(tab)  # the mark is fresh; nothing has changed yet
    calls: list[int] = []
    _counting_diff(monkeypatch, calls)

    enabled = inker_sheet.can_propagate(state, tab)
    reason = inker_sheet.propagate_reason(state, tab)

    assert not enabled
    assert reason == inker_sheet.NO_MARK
    assert len(calls) == 1, "can_propagate and propagate_reason diffed the cel twice"


def test_a_new_tick_after_a_real_edit_still_sees_it(monkeypatch):
    state, tab = _scene()
    inker_sheet.sync_mark(tab)
    calls: list[int] = []
    _counting_diff(monkeypatch, calls)

    assert not inker_sheet.can_propagate(state, tab)

    anim = tab.doc.anim
    cel = anim.cels[(anim.tracks[0].uid, anim.frames[0].uid)]
    cel.pixels[10, 8] = (255, 0, 0, 255)

    # A new frame draws: ``draw_strip`` calls ``sync_mark`` again before
    # asking anything, so the stale within-tick cache above must not answer.
    inker_sheet.sync_mark(tab)
    assert inker_sheet.can_propagate(state, tab)
    assert len(calls) == 2, "the post-edit check should have run its own diff"


def test_a_second_press_after_propagate_remarks_sees_no_mark(monkeypatch):
    """A cache keyed on the (track, frame) pair rather than the mark's own
    identity would survive ``propagate``'s re-mark on the same cell and keep
    answering for a patch that was already sent."""
    state, tab = _scene()
    inker_sheet.sync_mark(tab)

    anim = tab.doc.anim
    cel = anim.cels[(anim.tracks[0].uid, anim.frames[0].uid)]
    cel.pixels[10, 8] = (255, 0, 0, 255)

    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        toast=lambda *a, **k: None,
    )
    from realmspinner.studio.modes.inker import ops as inker_ops

    assert inker_ops.run(ctx, inker_ops.get("sheet_propagate"))
    # Same tick as the successful press (no new frame drawn), but the mark
    # was refreshed inside ``propagate`` -- a second press must see nothing.
    assert not inker_ops.run(ctx, inker_ops.get("sheet_propagate"))
