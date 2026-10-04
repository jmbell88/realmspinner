"""Regression tests for the 2026-09-26 audit, findings create-workspace-06
and create-workspace-07, both in ``studio/modes/create/ui/workspace.py``.
"""

from __future__ import annotations

from realmspinner.studio.modes.create.ui import workspace as gw

# --- create-workspace-07: count_line pluralises "generation" ---------------


def test_count_line_pluralises_generations_for_more_than_one():
    plan = gw.Plan(candidates=1, generations=4, duration="a few seconds", stages="x", recipe="y")
    assert plan.count_line == "1 candidate · 4 image generations"


def test_count_line_keeps_generation_singular_for_exactly_one():
    plan = gw.Plan(candidates=1, generations=1, duration="a few seconds", stages="x", recipe="y")
    assert plan.count_line == "1 candidate · 1 image generation"


# --- create-workspace-06: the tray never walks the whole cache ---------------
#
# ``_recent_results`` (an ``islice`` over ``ctx.cache.jobs``) was the first fix
# here and has since been replaced by ``families.results`` over the memoised
# index; it was deleted when nothing called it (2026-10-03 audit, finding
# create-46). What the old test guarded -- a frame never walks every cached job
# to draw three cards -- is asserted on the functions the tray really calls.


def test_the_tray_reads_the_memoised_index_and_never_walks_the_cache():
    import inspect

    for fn in (gw.draw, gw.should_draw):
        source = inspect.getsource(fn)
        assert "ctx.cache.jobs" not in source, fn.__name__
        assert "families.results(" in source or "session.index" in source, fn.__name__
