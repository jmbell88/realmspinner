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


def test_chapter_07_does_not_deny_the_decimate_and_retopology_clay_ships():
    """Chapter 7's Make 3D section told the reader outright that "there is no
    decimate or retopology operation" in Clay and sent them downstream to the
    retarget panel. Both shipped in ``d416cb42`` as fully wired background ops
    with their own registry rows, hints and params, and chapter 30 documents
    them correctly -- so the getting-started chapter was denying two features
    a reader can press today.

    The truth is read from the op registry rather than restated here.
    """
    ops = CLAY_OPS.read_text(encoding="utf-8")
    assert 'name="decimate"' in ops
    assert 'name="retopo"' in ops

    text = _flat(_chapter("07-modelling.md"))
    assert "there is no decimate or retopology operation" not in text
    assert "30-clay.md" in text


# --- clay-31: chapter 30's Godot collider suffix ---------------------------


def test_chapter_30s_godot_collider_suffix_is_the_one_the_exporter_emits():
    """Chapter 30 gave Godot's collider suffix as ``-colonly``.
    ``_godot_collider_name`` only ever emits ``-convcolonly``, whatever the
    collider kind, because every kind Clay fits is convex -- its own comment
    says so. A reader following the manual would look for a name the exporter
    never writes.

    ``-colonly`` is a substring of ``-convcolonly``, so the check is on the
    backticked spelling the manual uses, not on a bare substring.
    """
    engines = (KERNELS / "mesh" / "engines.py").read_text(encoding="utf-8")
    match = re.search(r'return f"\{mesh_name\}_\{index:02d\}(-\w+)"', engines)
    assert match, "could not find _godot_collider_name's emitted suffix"
    suffix = match.group(1)
    assert suffix == "-convcolonly", suffix

    text = _flat(_chapter("30-clay.md"))
    assert f"a `{suffix}` suffix for Godot" in text
    assert "a `-colonly` suffix for Godot" not in text
