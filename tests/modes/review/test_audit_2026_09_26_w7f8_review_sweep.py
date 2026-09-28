"""Regression tests for the 2026-09-26 audit's Review findings shell-review-
settings-03 through -06 (wave w7f8).

Each test's name is the claim; each failed against the code as the audit
found it and passes against the fix in ``studio/modes/review/mode.py`` and
``studio/modes/review/ui/workspace.py``.
"""

from __future__ import annotations

import pytest

from modes.review.test_review_mode import FakeCtx, _Done, _mesh, _scanned, _sweep, _two_sweeps
from realmspinner.studio.modes.review import mode as review_mode


@pytest.fixture
def ctx(svc):
    return FakeCtx(svc)

# --- shell-review-settings-03 -------------------------------------------------


def test_a_rejected_axis_value_is_recorded_as_a_field_error_not_only_a_toast(ctx, svc):
    """``launch`` used to toast a ``validate_sweep`` refusal's ``field`` and
    forget it -- nothing was ever recorded on ``ctx.state``, so no control
    could ever be rung. It now also calls ``note_field_error``, the same door
    ``settings_2d``/``sheet_panel`` already use for a service refusal that
    names a field."""
    form = review_mode.ensure(ctx).form
    form.prompt = "a chest"
    form.seeds = "1"
    form.axes = [{"param": "trellis_band", "values": "8, 999"}]

    assert review_mode.launch(ctx) is False
    # The toast stays (a sweep script has no control for the auditor to look
    # at), but the field is now also addressable.
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "trellis_band" in ctx.state.field_errors
    assert ctx.state.field_errors["trellis_band"]


def test_an_empty_prompt_rings_the_prompt_field(ctx):
    """``_validate`` refuses a blank prompt with ``field="prompt"`` -- the one
    field this form actually draws a control for -- and the refusal must land
    there."""
    form = review_mode.ensure(ctx).form
    form.prompt = ""
    form.axes = [{"param": "trellis_band", "values": "8"}]
    form.seeds = "1"

    assert review_mode.launch(ctx) is False
    assert ctx.state.field_errors.get("prompt")


# --- shell-review-settings-04 -------------------------------------------------


def test_a_refused_open_labels_leaves_the_previous_pass_in_place(ctx):
    """``open_labels`` used to swap in a fresh, empty ``LabelPass`` *before*
    submitting, and a refused submit left that orphaned pass in place -- the
    pane then read "nothing left to label" forever, with no task in flight to
    ever refill it. A refused press must change nothing."""
    refused = FakeCtx(ctx.svc, accept=False)
    state = review_mode.ensure(refused)
    previous = review_mode.LabelPass(stage="reference")
    previous.rows = [{"job_id": "kept", "prompt": "x", "image": None, "verdict": None}]
    state.labels = previous

    review_mode.open_labels(refused, "blank")

    assert state.labels is previous, "the refused press must not orphan the running pass"
    assert state.labels.rows == previous.rows


def test_a_refused_open_labels_with_no_previous_pass_leaves_none(ctx):
    """The symmetric case: no pass was running, so "roll back" means back to
    nothing, not to an empty pass that looks like one."""
    refused = FakeCtx(ctx.svc, accept=False)
    state = review_mode.ensure(refused)
    assert state.labels is None

    review_mode.open_labels(refused, "blank")

    assert state.labels is None


def test_an_accepted_open_labels_still_swaps_in_the_new_pass(ctx):
    """The fix above must not break the ordinary path: a submit that is
    accepted still shows the new question's pass."""
    state = review_mode.ensure(ctx)
    state.labels = review_mode.LabelPass(stage="reference")

    review_mode.open_labels(ctx, "blank")

    assert state.labels is not None
    assert state.labels.stage == "blank"


# --- shell-review-settings-05 -------------------------------------------------


def test_regrading_the_same_unit_in_one_pass_does_not_inflate_filed(ctx, svc):
    """"21 of 20": grading a unit, stepping back to it (the toast's own "Left
    arrow to re-grade it"), and grading it again used to count it twice
    against ``judging.filed`` and again against accepted/rejected."""
    _mesh(svc, "a chest")
    _mesh(svc, "a sword")
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    assert state.judging is not None
    total = state.judging.total

    review_mode.record(ctx, 3)  # accept the first unit
    assert state.judging.filed == 1
    assert state.judging.accepted == 1

    review_mode.step(state, -1)  # back onto the unit just graded
    review_mode.record(ctx, 3)  # the exact same verdict, re-filed

    assert state.judging.filed == 1, "a re-grade of the same row must not refile it"
    assert state.judging.accepted == 1
    assert state.judging.total == total


def test_regrading_with_a_different_verdict_moves_the_bucket_without_refiling(ctx, svc):
    """A re-grade that changes its mind (accept -> reject) must move the unit
    between the accepted/rejected buckets rather than adding to both."""
    _mesh(svc, "a chest")
    _mesh(svc, "a sword")
    state = _scanned(ctx)
    review_mode.start_judging(ctx)

    review_mode.record(ctx, 3)  # accept
    assert state.judging.filed == 1
    assert state.judging.accepted == 1
    assert state.judging.rejected == 0

    review_mode.step(state, -1)
    review_mode.record(ctx, -3)  # reject the same unit instead

    assert state.judging.filed == 1, "still one unit filed, not two"
    assert state.judging.accepted == 0
    assert state.judging.rejected == 1


def test_relabelling_the_same_row_does_not_inflate_the_label_count(ctx, svc):
    """The image-label sibling of the same bug: ``record_label`` used to add a
    second positive or negative for a row a reviewer stepped back onto and
    labelled again."""
    ids = [_mesh(svc, "a"), _mesh(svc, "b")]
    ctx.state.review = review_mode.ReviewState()
    labels = review_mode.LabelPass(stage="blank")
    labels.rows = [
        {"job_id": ids[0], "prompt": "a", "image": None, "verdict": None},
        {"job_id": ids[1], "prompt": "b", "image": None, "verdict": None},
    ]
    ctx.state.review.labels = labels

    assert review_mode.record_label(ctx, "accept") is True
    assert labels.status.get("positives") == 1
    assert labels.status.get("labels") == 1

    review_mode.advance_labels(labels)
    labels.index = 0  # back onto the row just labelled
    assert review_mode.record_label(ctx, "accept") is True

    assert labels.status.get("positives") == 1, "a re-label of the same row must not recount"
    assert labels.status.get("labels") == 1


def test_relabelling_with_a_different_verdict_moves_the_label_bucket(ctx, svc):
    job_id = _mesh(svc, "a")
    ctx.state.review = review_mode.ReviewState()
    labels = review_mode.LabelPass(stage="blank")
    labels.rows = [{"job_id": job_id, "prompt": "a", "image": None, "verdict": None}]
    ctx.state.review.labels = labels

    assert review_mode.record_label(ctx, "accept") is True
    assert labels.status.get("positives") == 1
    assert labels.status.get("negatives", 0) == 0
    assert labels.status.get("labels") == 1

    labels.index = 0
    assert review_mode.record_label(ctx, "reject") is True

    assert labels.status.get("positives") == 0
    assert labels.status.get("negatives") == 1
    assert labels.status.get("labels") == 1, "still one row labelled, not two"


# --- shell-review-settings-06 -------------------------------------------------


def test_a_scan_started_before_a_cleanup_does_not_overwrite_what_the_cleanup_did(ctx, svc):
    """A rescan (``SCAN_KEY``) already in flight when a cleanup lands used to
    apply its (now stale) results over whatever the cleanup just did, because
    the cleanup's own rescan request was refused outright (``scanning`` was
    still true) and never asked again."""
    newer, older = _two_sweeps(svc)
    state = _scanned(ctx)

    # A rescan starts, but has not "come back" yet -- ``ctx.result`` from this
    # submit is what a real task would eventually deliver.
    review_mode.scan(ctx)
    stale_result = ctx.result
    assert state.scanning is True

    # A cleanup lands while that scan is still in flight.
    review_mode.on_task_done(
        ctx, _Done(review_mode.CLEANUP_KEY, {"ok": True, "deleted": 2, "remaining": 0, "kept": 0})
    )
    # Its own rescan request was refused (one was already running), so no
    # second SCAN_KEY was actually submitted underneath it.
    assert state.scanning is True

    # The original, now-stale scan result arrives.
    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, stale_result))

    # It must not have been applied as the final answer -- it was collected
    # before the cleanup ran, under the old generation.
    assert ctx.submitted[-1] == review_mode.SCAN_KEY, "a fresh scan was asked for instead"
    assert state.scanning is True, "the fresh scan it asked for is still in flight"


def test_the_generation_token_advances_once_per_cleanup_landing(ctx, svc):
    """The token this fix hinges on: every cleanup/delete/remove completion
    bumps it, exactly once, so a scan launched before it can be told apart
    from one launched after."""
    _sweep(svc)
    state = _scanned(ctx)
    before = state.scan_generation

    review_mode.on_task_done(
        ctx, _Done(review_mode.DELETE_KEY, {"ok": True, "deleted": 1, "remaining": 0, "kept": 0})
    )

    assert state.scan_generation == before + 1
