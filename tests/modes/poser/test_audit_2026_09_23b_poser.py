"""Regression tests for the 2026-09-23 (second run) audit's poser-03
(studio/modes/poser/mode.py), poser-06 (characters/family.py), poser-07
(studio/modes/poser/engine/qa.py) and poser-08
(studio/modes/poser/ui/panes/send.py).

``tests/modes/poser/test_poser_mode.py`` already carries the fixture
apparatus ``select_key``'s own tests need (``FakeCtx``, ``FakeViewer``,
``_bound_viewer``, ``_clip_library``); reused here by import rather than
duplicated, the way ``tests/`` mirrors the source tree but does not
re-derive its own scaffolding per file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from realmspinner.characters import family
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.engine import qa
from realmspinner.studio.modes.poser.ui.panes import send as send_pane
from realmspinner.studio.modes.poser.ui.panes import sheet as sheet_pane


def _sibling(name: str):
    # Loaded by path, not as ``tests.modes.poser.<name>``: a maintainer checkout
    # also carries ``dev/tests``, which shadows the ``tests`` package name in the
    # full run and made the dotted import fail at collection.
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_poser_sibling_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tpm = _sibling("test_poser_mode")
tqa = _sibling("test_qa")

# --- poser-03 ------------------------------------------------------------


def test_apply_key_folds_the_root_offset_into_the_same_undo_step_as_the_bones():
    """``apply_key`` -> ``select_key`` used to call ``editor.apply_preset``
    (undoable, pushes its own step) and then ``editor.set_root_translation``
    with no ``record()`` bracket at all -- so the root offset landed on the
    live armature with no history entry of its own. Redoing the
    ``apply_preset`` step then restored the *pre-offset* snapshot it
    recorded: bones back, root offset lost.
    """
    lib = tpm._clip_library()
    # Give key "B" (the clip's middle key) a root offset to carry.
    lib["poses"][1]["root_translation"] = [0.2, 0.0, 0.0]
    ctx = tpm.FakeCtx()
    ctx.poser_viewer = tpm._bound_viewer()
    state = poser_mode.ensure(ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(ctx, lib)
    editor = ctx.poser_viewer.editor

    poser_mode.select_key(ctx, 1)
    assert editor.root_translation() == pytest.approx([0.2, 0.0, 0.0])
    posed_bones = editor.pose()

    assert editor.undo() is True
    assert editor.redo() is True

    # The bug: redo restored the bones (apply_preset's own step) but left the
    # root offset at zero, because it was never part of any pushed step.
    assert editor.root_translation() == pytest.approx([0.2, 0.0, 0.0])
    assert editor.pose()["spine"] == pytest.approx(posed_bones["spine"])


# --- poser-06 --------------------------------------------------------------


def test_archetype_keys_matches_every_archetype_the_registry_ships():
    """``ARCHETYPE_KEYS`` used to hand-duplicate ``_ARCHETYPES``' own keys
    with nothing pinning the two together, so a fifth archetype's registry
    row could ship with this vocabulary table still naming only the first
    four."""
    assert tuple(family.archetypes()) == family.ARCHETYPE_KEYS
    assert set(family.ARCHETYPE_KEYS) == set(family.archetypes())
    for key in family.ARCHETYPE_KEYS:
        # Every advertised key must actually resolve -- get_archetype raises
        # CharacterError on a key the registry does not carry.
        family.get_archetype(key)


# --- poser-07 ----------------------------------------------------------------


def test_sheet_score_worst_can_name_a_warn_level_cell_with_no_bad_cell_present():
    """``SheetScore.worst`` is documented (after the fix) as the single
    worst-scoring flagged cell, which may be a warn-level one -- not, as the
    docstring used to claim, always a cell past its *bad* threshold. A
    one-shot two-frame "walk" shifted just enough to cross ``shape_delta``'s
    warn threshold (0.35) but stay under its bad one (0.60) -- built the same
    way ``test_qa.py``'s own shift fixtures are -- produces exactly that:
    ``worst`` names the shifted cell and every cell still scores at most
    ``LEVEL_WARN``. This pins the existing (correct) behaviour; the fix here
    was to the docstring alone, so this test passes identically before and
    after it.
    """
    layout = tqa._layout([("walk", 2, ("front",), False)])

    def paint(index, crop):
        tqa._figure(crop, shift=1 if index == 1 else 0)

    score = tqa._score(layout, paint)
    warn, bad = qa.THRESHOLDS["shape_delta"]
    moved = score.lookup()[("walk", "front", 1)]
    assert warn <= moved.metrics["shape_delta"] < bad
    assert all(qa.level(cell) != qa.LEVEL_BAD for cell in score.cells)
    assert score.worst is not None
    assert score.worst[0] == "shape_delta"
    assert score.worst[2] == 1


# --- poser-08 ----------------------------------------------------------------


def test_send_pane_camera_helper_is_the_sheet_panes_helper():
    """The 2026-09-23 audit, finding poser-08: ``send.py`` used to carry its
    own byte-identical copy of ``sheet.py``'s ``_camera_helper``, untested
    and free to drift. Now shared."""
    assert send_pane._camera_helper is sheet_pane._camera_helper
