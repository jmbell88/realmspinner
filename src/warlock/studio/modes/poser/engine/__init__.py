"""Troupe: the headless character-sheet engine.

Imports no imgui, moderngl, pygame or ``service`` -- pinned by
``tests/modes/poser/test_poser_engine_imports.py``, the ``tests/modes/inker/test_sheetout.py``
rule at its fourth instance.
"""

from __future__ import annotations

from . import qa, spec, ulpc

__all__ = ["qa", "spec", "ulpc"]
