"""Regression tests for the 2026-09-26 audit, findings create-workspace-06
and create-workspace-07, both in ``studio/modes/create/ui/workspace.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio.modes.create.ui import workspace as gw

# --- create-workspace-07: count_line pluralises "generation" ---------------


def test_count_line_pluralises_generations_for_more_than_one():
    plan = gw.Plan(candidates=1, generations=4, duration="a few seconds", stages="x", recipe="y")
    assert plan.count_line == "1 candidate · 4 image generations"


def test_count_line_keeps_generation_singular_for_exactly_one():
    plan = gw.Plan(candidates=1, generations=1, duration="a few seconds", stages="x", recipe="y")
    assert plan.count_line == "1 candidate · 1 image generation"


# --- create-workspace-06: _recent_results stops once it has three ----------


class _CountingJobs:
    """Wraps a job list and counts how many entries a consumer actually
    pulled, so a fix that stops early is distinguishable from one that scans
    everything and throws the rest away."""

    def __init__(self, jobs):
        self._jobs = jobs
        self.yielded = 0

    def __iter__(self):
        for job in self._jobs:
            self.yielded += 1
            yield job


def _done_job(job_id):
    return {"id": job_id, "status": "done"}


def test_recent_results_stops_once_three_matches_are_found():
    jobs = _CountingJobs([_done_job(str(i)) for i in range(50)])
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=jobs))

    result = gw._recent_results(ctx)

    assert len(result) == gw._RESULT_COLUMNS == 3
    assert jobs.yielded == 3, (
        "the tray only ever shows three cards, so building it must not walk "
        f"every cached job -- it pulled {jobs.yielded} of 50"
    )
