"""The 2026-10-14 audit, cleanup round C1: the character agent surface's leftover doors.

``eed32549`` removed the MCP character tools and Poser, and left their service doors behind with
no production caller (findings poser-characters-07 and poser-jobs-02). Each name below is
asserted absent, and each survivor is asserted to keep the one caller it has.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from realmspinner.service import characters, export, pixelopts, rig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "realmspinner"


def test_the_character_agent_doors_are_gone():
    for name in (
        "ASSET_FILTERS",
        "_CHARACTER_ASSET_PAGE",
        "_CHARACTER_ASSET_SCAN_CAP",
        "reroll_character",
        "_character_asset_template",
        "_character_asset_row",
        "list_character_assets",
        "character_job",
        "agent_export_stem",
        "_refuse_foreign_folder",
        "_plan",
    ):
        assert not hasattr(characters, name), f"service.characters.{name}"
    for name in (
        "run_character_export",
        "CHARACTER_EXPORTS",
        "CharacterExport",
        "_export_animated_glb",
        "_export_godot_scene",
    ):
        assert not hasattr(export, name), f"service.export.{name}"


def test_export_godot_keeps_only_the_parameters_its_one_caller_passes():
    """The inspector's "Export for Godot..." passes ``dest_dir``; ``stem`` and ``overwrite``
    were an agent's way to name or replace a folder."""
    assert list(inspect.signature(characters.export_godot).parameters) == [
        "svc",
        "job_id",
        "dest_dir",
    ]
    # ``_free_stem`` stays: ``export_godot`` is how a repeat export lands beside the first.
    assert "suffixes" not in inspect.signature(characters._free_stem).parameters


def test_the_character_dry_run_and_options_carry_no_key_nothing_reads(svc):
    assert "default_recipe" not in characters.character_options(svc)
    plan = characters.recipe_from_prompt(svc, "a fire ogre")
    assert "cells" not in plan
    assert set(plan) == {"recipe", "resolution", "ignored", "estimate_minutes"}


def test_the_character_effects_module_is_gone():
    """``characters/effects.py`` drew a flame onto a character sheet's cells; the sheet left with
    Poser, and nothing in ``src/`` imported it (finding poser-characters-05)."""
    assert importlib.util.find_spec("realmspinner.characters.effects") is None
    assert not (SRC / "characters" / "effects.py").exists()
    assert not (ROOT / "tests" / "characters" / "test_effects.py").exists()


def test_the_service_rig_module_has_no_skeleton_editor_door_or_refuse_existing():
    """``edit_skeleton``, ``limb_presets`` and ``create_rig(refuse_existing=)`` served the Poser
    skeleton editor and the character agent surface; neither exists."""
    assert not hasattr(rig, "edit_skeleton")
    assert not hasattr(rig, "limb_presets")
    assert "refuse_existing" not in inspect.signature(rig.create_rig).parameters


def test_pixelopts_has_no_size_range_and_no_value_range():
    assert "size_range" not in inspect.signature(pixelopts.check_pixel_options).parameters
    assert "value_range" not in inspect.signature(pixelopts._on_the_ladder).parameters


def test_the_shipped_clip_libraries_do_not_call_themselves_troupes():
    for name in ("humanoid", "bird", "blob", "quadruped"):
        text = (SRC / "templates" / "clips" / f"{name}.json").read_text(encoding="utf-8")
        assert "Troupe" not in text, name
