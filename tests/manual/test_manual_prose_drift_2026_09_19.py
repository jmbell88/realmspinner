"""Documentation findings from the 2026-09-19 ``clay`` audit, pinned against the tree.

Each test reads its truth from the module that makes the claim true or false
rather than repeating a second hand-written copy, the same shape as
``test_manual_promises.py`` and the two earlier prose-drift files. Kept
separate from those for the same reason they are kept separate from each
other: they belong to different fixers' file lists.

Both findings here are the same defect in two chapters -- ``d416cb42``
("Clay becomes a game-asset modeller") shipped operations and naming that the
manual still describes as they were before it landed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "manual"
KERNELS = ROOT / "src" / "realmspinner" / "kernels"
CLAY_OPS = ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "ops.py"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Prose with hand-wrapped line breaks collapsed to single spaces, so a
    phrase that happens to straddle two lines in the manual's own wrapping
    still matches a plain substring check."""
    return re.sub(r"\s+", " ", text)


# --- clay-08: chapter 7 denied two shipped operations ----------------------


def test_chapter_07_does_not_promise_the_decimate_and_retopology_clay_dropped():
    """Chapter 7's Make 3D section once told the reader "there is no decimate or
    retopology operation" in Clay while both shipped (``d416cb42``), and the fix
    was to document them. The picoCAD cut removed them again, so the claim now
    runs the other way: the chapter must not send a reader to a Decimate or a
    Retopologize button, and must still point at chapter 30 for what Clay does.

    The truth is read from the op registry rather than restated here.
    """
    ops = CLAY_OPS.read_text(encoding="utf-8")
    assert 'name="decimate"' not in ops
    assert 'name="retopo"' not in ops

    text = _flat(_chapter("07-modelling.md"))
    assert "there is no decimate or retopology operation" not in text
    assert "Retopologize" not in text
    assert "Decimate" not in text
    assert "30-clay.md" in text


# --- clay-31: chapter 30 and the engine collider names ---------------------


def test_chapter_30_names_no_collider_or_engine_profile_the_exporter_no_longer_writes():
    """Chapter 30 gave Godot's collider suffix (``-colonly``, then ``-convcolonly``)
    and described an export-engine setting that renamed colliders on the way out.
    ``kernels/mesh/engines.py`` and every collider are gone: an OBJ is written
    plain, so a reader must not be told to look for ``UCX_`` or ``-convcolonly``
    names, or to pick an export engine.
    """
    assert not (KERNELS / "mesh" / "engines.py").exists()

    text = _flat(_chapter("30-clay.md"))
    for stale in ("-convcolonly", "-colonly", "UCX_", "export engine", "Game check"):
        assert stale not in text, f"chapter 30 still mentions {stale!r}"
