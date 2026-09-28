"""The Song file panel's draw order, read from source.

The 2026-09-18 audit (finding sirens-03) found *Closeness* drawn under
*Export audio*, the button it has nothing to do with, while its own docstring
and chapter 36 put it under *Compose in Muse...*, the hand-off it governs.
"""

from __future__ import annotations

import ast
import inspect

from realmspinner.studio.modes.sirens.ui.panes import bridge


def test_closeness_is_drawn_under_compose_in_muse_not_under_export_audio() -> None:
    src = inspect.getsource(bridge._export)
    compose = src.index("Compose in Muse...")
    closeness = src.index("_closeness(ctx)")
    assert closeness > compose, (
        "Closeness governs Compose in Muse; draw it under that button, not under Export audio"
    )


def test_closeness_does_not_reach_into_muses_results_pane_for_its_table() -> None:
    """The 2026-09-26 audit, finding sirens-panes-03: ``_closeness`` used to
    ``from ....muse.ui.panes.results import DERIVE_FIELDS`` and index it by
    ``"ref_audio_strength"`` -- a cross-mode reach onto a sibling module's
    internal data table, function-scoped only enough to dodge
    ``tests/test_layering.py``'s module-scope-only AST walk, not the
    documented "call a door by name" escape hatch. It should use its own
    vendored ``_CLOSENESS`` tuple instead, and no import anywhere in the
    module -- module- or function-scoped -- should still reach Muse's results
    pane or name ``DERIVE_FIELDS``. Walked with ``ast`` rather than a plain
    substring check on the source, so an explanatory comment mentioning the
    old import (as this fix's own docstring does) cannot make the test lie.
    """
    tree = ast.parse(inspect.getsource(bridge))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").endswith("muse.ui.panes.results"), node.module
            assert all(alias.name != "DERIVE_FIELDS" for alias in node.names)
    closeness_src = inspect.getsource(bridge._closeness)
    assert "_CLOSENESS" in closeness_src


def test_closeness_tuple_has_the_label_range_and_hint_shape() -> None:
    label, low, high, hint = bridge._CLOSENESS
    assert label == "Closeness"
    assert low == 0.0
    assert high == 0.9
    assert isinstance(hint, str) and hint
