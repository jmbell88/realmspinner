"""The Song file panel's draw order, read from source.

The 2026-09-18 audit (finding sirens-03) found *Closeness* drawn under
*Export audio*, the button it has nothing to do with, while its own docstring
and chapter 36 put it under *Compose in Muse...*, the hand-off it governs.
"""

from __future__ import annotations

import inspect

from realmspinner.studio.modes.sirens.ui.panes import bridge


def test_closeness_is_drawn_under_compose_in_muse_not_under_export_audio() -> None:
    src = inspect.getsource(bridge._export)
    compose = src.index("Compose in Muse...")
    closeness = src.index("_closeness(ctx)")
    assert closeness > compose, (
        "Closeness governs Compose in Muse; draw it under that button, not under Export audio"
    )
