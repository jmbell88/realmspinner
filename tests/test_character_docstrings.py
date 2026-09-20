"""Docstrings that name a specific module, function or caller are claims a
test can check -- the 2026-09-14 audit found two that had drifted from the
code beside them (troupe-03, troupe-04). This file is a home for that kind
of check across the character pipeline, so the next drifted docstring has
somewhere to land instead of a fresh top-level test module.
"""

from __future__ import annotations

import inspect

from realmspinner.kernels import charsheet
from realmspinner.service import characters as svc_characters


def test_export_godot_docstring_names_the_module_that_actually_defines_rename_animations():
    """troupe-03: ``export_godot``'s docstring said ``godotscene.rename_animations``
    runs on a copy of the exported bytes, but ``export_godot`` itself calls
    ``glbio.rename_animations`` -- ``godotscene.__all__`` has no such function
    at all."""
    from realmspinner import godotscene

    assert not hasattr(godotscene, "rename_animations")
    doc = inspect.getdoc(svc_characters.export_godot) or ""
    assert "glbio.rename_animations" in doc
    assert "godotscene.rename_animations" not in doc


def test_movement_min_frames_docstring_does_not_claim_callers_that_do_not_exist():
    """troupe-04: ``movement_min_frames``'s docstring claimed
    ``characters/recipe.py`` and ``service/troupe.py`` "still index"
    ``MOVEMENT_MIN_FRAMES`` directly -- both already call the function
    instead, and nothing in ``src/`` indexes the dict directly any more."""
    doc = inspect.getdoc(charsheet.movement_min_frames) or ""
    assert "still index" not in doc
    assert "callers that still index it directly" not in doc
