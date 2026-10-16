"""Candidates come up best-match first, and the best is offered as the selection.

This reverses ``candidates_panel``'s old "no ordering, no pre-selection" rule for mesh
candidates (2026-10-15). The order is ``kernels.mesh.score``'s sort key and the offer moves
the selection only: Keep is still a click, nothing is hidden, and a click of the user's
own is never undone.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import score
from realmspinner.studio import candidates as cand
from realmspinner.studio import widgets
from realmspinner.studio.modes.create.ui import workspace
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.state import AppState


def _score(iou, **kw):
    base = {
        "iou": iou, "best_view": 0, "holes": 0.0, "hole_max": 0.05, "components": 1,
        "nonmanifold": 0, "boundary": 0, "budget_ratio": 1.0, "budget_miss": None,
    }
    base.update(kw)
    return score.build(**base)


def _row(i: int, iou: float | None, *, status="done", group="g1", workspace_id="w1", **kw):
    params: dict[str, Any] = {"create_workspace": workspace_id, "mesh_seed": 100 + i}
    if iou is not None:
        params["mesh_score"] = _score(iou, **kw)
    return {
        "id": f"m{i}", "kind": "image", "stage": "model", "status": status,
        "candidate_group": group, "candidate_index": i, "created_at": 100.0 + i * 0.01,
        "prompt": "a chest", "params": params,
    }


def _ids(rows):
    return [r["id"] for r in rows]


# --- the order ------------------------------------------------------------------


def test_candidates_sorted_by_score_then_index():
    # Created #0..#3. #2 matches best, #1 and #3 tie and keep their seed order, and the
    # unscored #0 (still running) comes last.
    rows = [
        _row(0, None, status="running"), _row(1, 0.80), _row(2, 0.95), _row(3, 0.80),
    ]
    assert _ids(cand.batch(rows, "g1").members) == ["m2", "m1", "m3", "m0"]


def test_a_picker_group_is_ordered_the_same_way():
    rows = [
        _row(0, 0.5, workspace_id="x"), _row(1, 0.9, workspace_id="x"),
    ]
    # ``pending`` skips a Create workspace's batch (ordinary work, no picker)...
    assert cand.pending(rows) is None
    # ...and orders one that is not.
    for row in rows:
        row["params"].pop("create_workspace")
    assert _ids(cand.pending(rows).members) == ["m1", "m0"]


def test_a_demoted_mesh_sorts_after_a_plainer_one_that_met_its_budget():
    rows = [
        _row(0, 0.99, budget_miss={"requested": 5000, "achieved": 13821}),
        _row(1, 0.60),
    ]
    assert _ids(cand.batch(rows, "g1").members) == ["m1", "m0"]


def test_rank_batches_reorders_inside_a_batch_and_leaves_everything_else_where_it_was():
    other = {"id": "r1", "kind": "text", "stage": "reference", "status": "done", "params": {}}
    older = [_row(0, 0.9, group="g0"), _row(1, 0.1, group="g0")]
    older[0]["id"], older[1]["id"] = "o0", "o1"
    newer = [_row(0, 0.2), _row(1, 0.9)]
    rows = [newer[0], other, newer[1], older[0], older[1]]
    ranked = cand.rank_batches(rows)
    # The batch's own two slots (0 and 2) swap; the unrelated row and the older batch stay.
    assert _ids(ranked) == ["m1", "r1", "m0", "o0", "o1"]
    assert _ids(rows) == ["m0", "r1", "m1", "o0", "o1"], "the input list is not mutated"


def test_rank_batches_with_no_group_is_a_no_op():
    rows = [{"id": "a", "params": {}}, {"id": "b", "params": {}}]
    assert cand.rank_batches(rows) == rows


def test_best_needs_a_finished_member_with_a_real_score():
    assert cand.best(cand.batch([_row(0, None), _row(1, None)], "g1")) is None
    running = _row(0, 0.99, status="running")
    assert cand.best(cand.batch([running, _row(1, 0.4)], "g1"))["id"] == "m1"


# --- the offer ------------------------------------------------------------------


def _ctx(rows, *, selected, follow, stage="mesh") -> Any:
    state = AppState(mode="create")
    state.create.workspace = "creation:w1"
    state.create.stage = stage
    state.selected = selected
    state.create.follow_selection = follow
    return SimpleNamespace(
        state=state,
        cache=SimpleNamespace(get={r["id"]: r for r in rows}.get, jobs=rows, active=None),
    )


@pytest.fixture
def picked(monkeypatch):
    chosen: list[str] = []
    monkeypatch.setattr(library, "select", lambda c, i: (chosen.append(i), c.state.select(i)))
    return chosen


def test_best_candidate_preselected_once_and_never_steals_selection(picked):
    rows = [_row(0, 0.5), _row(1, 0.9), _row(2, 0.7)]
    group = cand.batch(rows, "g1")
    # The app had put the selection on the first candidate to land.
    ctx = _ctx(rows, selected="m0", follow="m0")

    assert workspace.preselect_best_candidate(ctx, group) is True
    assert ctx.state.selected == "m1"
    assert ctx.state.create.follow_selection == "m1"

    # Once: the user then clicks another candidate, and nothing offers m1 again.
    ctx.state.select("m2")
    assert workspace.preselect_best_candidate(ctx, group) is False
    assert ctx.state.selected == "m2"
    assert picked == ["m1"]


def test_a_click_before_the_group_settles_is_never_undone(picked):
    rows = [_row(0, 0.5), _row(1, 0.9)]
    group = cand.batch(rows, "g1")
    ctx = _ctx(rows, selected="m0", follow="m0")
    ctx.state.select("m0")  # a deliberate click on #0 ...
    ctx.state.create.follow_selection = "ref"  # ... which is not where the app last put it
    assert workspace.preselect_best_candidate(ctx, group) is False
    assert ctx.state.selected == "m0"
    assert picked == []


def test_nothing_is_offered_until_the_whole_group_has_settled(picked):
    rows = [_row(0, 0.5), _row(1, None, status="running")]
    ctx = _ctx(rows, selected="m0", follow="m0")
    assert workspace.preselect_best_candidate(ctx, cand.batch(rows, "g1")) is False
    assert "g1" not in ctx.state.create.preselected, "not spent: it may still happen"
    rows[1] = _row(1, 0.9)
    ctx = _ctx(rows, selected="m0", follow="m0")
    assert workspace.preselect_best_candidate(ctx, cand.batch(rows, "g1")) is True


def test_no_scores_means_no_basis_and_no_offer(picked):
    rows = [_row(0, None), _row(1, None)]
    ctx = _ctx(rows, selected="m0", follow="m0")
    assert workspace.preselect_best_candidate(ctx, cand.batch(rows, "g1")) is False
    assert picked == []


def test_a_user_on_another_stage_or_creation_is_not_pulled_back(picked):
    rows = [_row(0, 0.5), _row(1, 0.9)]
    group = cand.batch(rows, "g1")
    elsewhere = _ctx(rows, selected="m0", follow="m0", stage="reference")
    assert workspace.preselect_best_candidate(elsewhere, group) is False
    other = _ctx(rows, selected="m0", follow="m0")
    other.state.create.workspace = "creation:another"
    assert workspace.preselect_best_candidate(other, cand.batch(rows, "g1")) is False
    assert picked == []


def test_keep_still_requires_a_click(picked, monkeypatch):
    """The offer moves the selection and nothing else: it never keeps, discards, or edits a row."""
    from realmspinner.studio.panes import candidates_panel

    def boom(*a, **k):
        raise AssertionError("pre-selection must not decide the group")

    monkeypatch.setattr(candidates_panel, "keep", boom)
    monkeypatch.setattr(candidates_panel, "discard", boom)
    rows = [_row(0, 0.5), _row(1, 0.9)]
    before = [dict(r, params=dict(r["params"])) for r in rows]
    ctx = _ctx(rows, selected="m0", follow="m0")
    assert workspace.preselect_best_candidate(ctx, cand.batch(rows, "g1")) is True
    assert rows == before
    # Every member is still in the group the picker offers: nothing was hidden.
    for row in rows:
        row["params"].pop("create_workspace")
    assert len(cand.pending(rows).members) == 2


# --- the pill -------------------------------------------------------------------


class _Drawn(list):
    """The chip labels drawn, and how many ``same_line`` calls were spent."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[int] = []


@pytest.fixture
def chips(monkeypatch):
    drawn = _Drawn()
    monkeypatch.setattr(widgets, "_chip", lambda label, colour, fill: drawn.append(label))
    monkeypatch.setattr(widgets.imgui, "same_line", lambda *a, **k: drawn.lines.append(1))
    return drawn


def test_over_budget_is_a_red_pill_naming_both_numbers(chips):
    widgets.budget_pill({"budget_miss": {"requested": 5000, "achieved": 13821, "reason": "x"}})
    assert len(chips) == 1
    assert "Over budget: 13,821 of 5,000" in chips[0]


def test_a_shape_miss_inside_the_budget_does_not_claim_to_be_over_budget(chips):
    widgets.budget_pill({"budget_miss": {"requested": 5000, "achieved": 4990, "reason": "x"}})
    assert "Over budget" not in chips[0]
    assert "shape check" in chips[0]


def test_no_miss_draws_nothing_and_spends_no_same_line(chips):
    widgets.budget_pill({}, inline=True)
    widgets.budget_pill(None, inline=True)
    widgets.budget_pill({"budget_miss": "not a dict"}, inline=True)
    assert chips == [] and chips.lines == []
