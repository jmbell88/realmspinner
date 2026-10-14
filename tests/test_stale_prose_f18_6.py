"""Prose that named something the code no longer has (the 2026-10-10 audit's F18-6).

Each test's name is the claim, and each fails against the text it replaced.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"


def test_nothing_still_cites_the_create_brief_shows_gate_that_is_gone():
    from realmspinner.studio.modes.create.ui import brief as create_brief
    from realmspinner.studio.modes.muse.ui import brief as muse_brief
    from realmspinner.studio.shell import frame

    assert not hasattr(create_brief, "shows")
    assert "create_brief.shows`` gates" not in inspect.getsource(muse_brief)
    assert "no ``shows``" not in inspect.getsource(muse_brief)
    assert "create_brief.shows`` no" not in inspect.getsource(frame)


def test_manual_28_says_a_regeneration_is_one_at_a_time_across_documents():
    """``InkerState.inpaint_pending`` is one slot on the mode's state, not on a
    tab, and ``_open_inpaint`` refuses a second while it is set."""
    from realmspinner.studio.modes.inker import state as inker_state

    text = " ".join((MANUAL / "25-inker.md").read_text(encoding="utf-8").split())
    section = text.split("## Regenerating a selection")[1].split("## Animation")[0]
    assert "per document" not in section
    assert "not one per open drawing" in text
    fields = inker_state.InkerState.__dataclass_fields__
    assert "inpaint_pending" in fields
    assert "inpaint_pending" not in inker_state.InkerDoc.__dataclass_fields__


def test_manual_33_says_a_grid_pack_ignores_trim():
    from realmspinner.studio.modes.packwright.engine import layout

    text = " ".join((MANUAL / "30-packwright.md").read_text(encoding="utf-8").split())
    assert "**Trim** (MaxRects only)" in text
    assert "A **grid** pack ignores Trim" in text
    assert "A grid pack never trims" in inspect.getdoc(layout)


def test_the_ants_test_names_the_function_the_pane_calls():
    source = (ROOT / "tests" / "modes" / "inker" / "test_ants.py").read_text(encoding="utf-8")
    assert "inker_state.to_screen" not in source
