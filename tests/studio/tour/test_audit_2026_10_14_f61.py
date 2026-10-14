"""The 2026-10-14 audit, tour findings tour-tour-1-01 .. 1-06 (scripts.py copy).

Every one of these is a sentence a tour card tells the reader that the screen it
points at does not bear out; the tests read the copy and the drawing code side
by side.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from realmspinner.studio.tour import TOURS

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "realmspinner"


def _step(tour_key: str, step_id: str):
    tour = next(t for t in TOURS if t.key == tour_key)
    return next(s for s in tour.steps if s.id == step_id)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_character_type_step_promises_no_sprite_sheet_and_no_missing_seed():
    """tour-tour-1-01: a Character press ends at a rigged mesh, not a sheet, and
    "3/4 top down" resolves a camera nothing consumes."""
    body = _step("first-hour", "character-type").body.lower()
    assert "sprite sheet" not in body
    assert "3/4" not in body and "top down" not in body
    assert "rigged mesh" in body
    # The copy still says "no seed", so the Character column must draw none.
    column = (
        SRC / "studio" / "modes" / "create" / "ui" / "panes" / "settings_character.py"
    ).read_text(encoding="utf-8")
    if "no seed" in body:
        assert "seed_row(" not in column
    # The rig needs Blender, which the manual and the refusal both say.
    assert "blender" in body or "rigging pack" in body


def test_the_character_type_comment_does_not_say_character_needs_nothing_downloaded():
    """tour-tour-1-01: the comment called Character the one type that runs with
    no weights, but its rig needs Blender (the Rigging pack)."""
    source = (SRC / "studio" / "tour" / "scripts.py").read_text(encoding="utf-8")
    assert "is the one generation type that does too" not in _flat(source.replace("# ", ""))


def test_the_layer_step_does_not_promise_a_button_the_layers_pane_lacks():
    """tour-tour-1-02: the timeline window has no add-layer button; a layer is
    added by the Layer menu or Ctrl+Shift+N."""
    body = _step("inker-basics", "layer").body
    assert "button on this panel" not in body
    assert "Ctrl+Shift+N" in body
    assert "Layer" in body and "New layer" in body


def test_the_tours_docstring_does_not_claim_first_hour_finishes_without_weights():
    """tour-tour-1-04: on a fresh install the Create door is shut, so the tour
    stops at ``stages``; the docstring said every step runs with no weights."""
    source = (SRC / "studio" / "tour" / "scripts.py").read_text(encoding="utf-8")
    doc = _flat(ast.get_docstring(ast.parse(source)) or "")
    assert "every step of it runs on a machine with no GPU and no weights" not in doc
    assert "stops" in doc and "Create" in doc and "weights" in doc


def test_the_colour_step_places_the_mode_switch_where_the_pane_draws_it():
    """tour-tour-1-03: the Colour/Indexed Mode row is inside the Palette header
    at the bottom of the panel, not "at the top"."""
    body = _step("inker-basics", "colour").body
    assert "at the top of this panel" not in body
    assert "Palette" in body
    colors = (
        SRC / "studio" / "modes" / "inker" / "ui" / "panes" / "colors.py"
    ).read_text(encoding="utf-8")
    assert '"Palette", default_open=bool(doc.palette)' in colors


def test_the_torch_steps_name_the_add_flyout_for_a_rail_too_short_for_quick_shapes():
    """tour-tour-1-06: a short viewport drops the four quick-shape buttons and
    leaves only "+" labelled Add."""
    for step_id in ("plate-add", "shaft", "flame-add"):
        body = _step("clay-torch", step_id).body
        assert "Add" in body and ("hidden" in body or "too short" in body), (step_id, body)
