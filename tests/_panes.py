"""Every dockable pane's source file, keyed by its pre-restructure pane name.

Restructure P5 started folding per-mode panes out of the flat
``studio/panes/`` directory into ``studio/modes/<mode>/ui/panes/`` (Clay
first; Inker and nine more modes will move the same way). A dozen tests in
this suite key their sweeps by filename or stem inside ``studio/panes/``
alone, and each such move silently drops the moved files out of every one
of those sweeps -- the sweep still runs, still passes, and just stops
covering the pane it used to cover. This module is the one place that
knows the current split between the flat directory and the per-mode
packages, so a sweep built on it keeps seeing a pane after it moves instead
of quietly losing it.
"""

from __future__ import annotations

from pathlib import Path

STUDIO = Path(__file__).resolve().parents[1] / "src/realmspinner/studio"

# Create's three settings panes were `settings_*.py` back when they lived in
# the flat `studio/panes/` directory, and were never `create_`-prefixed --
# P5 moved them into a mode package (`modes/create/ui/panes/`) without
# renaming them, so they keep their bare names here too rather than
# inventing a prefix that never existed.
_UNPREFIXED = {
    "create": {"settings_2d.py", "settings_3d.py", "settings_character.py"},
    "home": {"landing.py"},
    "library": {"library.py"},
    "settings": {"app_settings.py"},
}


def pane_files() -> dict[str, Path]:
    """Map each pane's pre-restructure filename to where it lives now.

    ``studio/panes/*.py`` panes keep their own name (``"inspector.py"``).
    ``studio/modes/<mode>/ui/panes/*.py`` panes are keyed
    ``f"{mode}_{name}"`` (``modes/clay/ui/panes/props.py`` ->
    ``"clay_props.py"``), matching the name they had before P5 moved them
    out of ``studio/panes/`` -- except the ``_UNPREFIXED`` set above, which
    never carried a mode prefix.

    ``panes/__init__.py`` is included (not a pane, but
    ``tests/manual/test_coverage.py`` has always swept it and names it in
    ``NO_HELP_BUTTON``); a moved package's own ``__init__.py`` is excluded,
    since nothing swept those before they existed.
    """
    files: dict[str, Path] = {}

    panes_dir = STUDIO / "panes"
    for path in sorted(panes_dir.glob("*.py")):
        files[path.name] = path

    modes_dir = STUDIO / "modes"
    for ui_panes in sorted(modes_dir.glob("*/ui/panes")):
        mode = ui_panes.parent.parent.name
        unprefixed = _UNPREFIXED.get(mode, set())
        for path in sorted(ui_panes.glob("*.py")):
            if path.name == "__init__.py":
                continue
            key = path.name if path.name in unprefixed else f"{mode}_{path.name}"
            files[key] = path

    # A broken search root (a renamed directory, a typo after the next mode
    # moves) would make this quietly sweep nothing instead of failing loudly
    # -- fail loudly instead.
    assert len(files) > 80, (
        f"pane_files() only found {len(files)} panes; its search roots "
        f"({panes_dir}, {modes_dir}/*/ui/panes) have probably drifted from "
        "where panes actually live -- fix the roots, not this floor"
    )
    return files


def pane_stems() -> dict[str, Path]:
    """Same mapping, keyed by stem (no ``.py``) -- what stem-keyed sweeps want."""
    return {name[:-3]: path for name, path in pane_files().items()}
