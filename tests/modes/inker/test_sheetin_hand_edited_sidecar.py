"""A hand-edited sidecar must open, not raise out of ``document_from_sheet``.

The 2026-10-10 audit's F18-1: ``int(entry.get("duration_ms") or 0)`` raised
``ValueError`` on ``"fast"`` (and ``OverflowError`` on a JSON ``Infinity``)
where ``Frame.__setattr__``'s ``clamp_duration`` has long treated both as "use
the default". The door that reads a sidecar owes the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.pixel import animation as anim_mod
from realmspinner.kernels.pixel.sheetin import document_from_sheet


def _sheet(frames):
    cell = 4
    atlas = np.zeros((cell, cell * 3, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    cells = [{"x": i * cell, "y": 0, "w": cell, "h": cell} for i in range(3)]
    anim = {"tags": [{"name": "walk_front", "start": 0, "end": 2, "loop": True}], "frames": frames}
    return document_from_sheet(atlas, cells, anim)


@pytest.mark.parametrize("bad", ["fast", "", "12ms", [80], {"ms": 80}, float("inf"), float("nan")])
def test_a_non_numeric_sidecar_duration_keeps_the_default_instead_of_raising(bad):
    doc = _sheet(
        [
            {"cell_index": 0, "duration_ms": bad},
            {"cell_index": 1, "duration_ms": 80},
            {"cell_index": 2, "duration_ms": 40},
        ]
    )
    assert doc.anim.frames[0].duration_ms == anim_mod.DEFAULT_DURATION_MS
    # The entries around the bad one are still read.
    assert doc.anim.frames[1].duration_ms == 80
    assert doc.anim.frames[2].duration_ms == 40


def test_a_numeric_string_duration_is_still_read():
    doc = _sheet([{"cell_index": 0, "duration_ms": "90"}])
    assert doc.anim.frames[0].duration_ms == 90


def test_a_non_numeric_cell_index_skips_its_entry_instead_of_raising():
    doc = _sheet(
        [
            {"cell_index": "first", "duration_ms": 80},
            {"cell_index": 1, "duration_ms": 80},
        ]
    )
    assert doc.anim.frames[0].duration_ms == anim_mod.DEFAULT_DURATION_MS
    assert doc.anim.frames[1].duration_ms == 80
