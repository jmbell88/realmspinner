"""Review's label loop, drawn: whether a disabled control explains itself.

``tests/test_review_mode.py`` covers ``review_mode`` (the headless half);
this file is for ``review_panes.py``'s own drawing defects, which only show up
once the buttons are actually submitted to imgui and the control census reads
them back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import probe
from warlock.studio.modes.review import mode as review_mode
from warlock.studio.modes.review.ui import workspace as review_panes


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw(imgui, ctx, state):
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 400.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    review_panes.ReviewPanes()._review_label_panel(ctx, state, review_mode)
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def test_the_good_button_explains_its_gate_like_its_two_neighbours(ui):
    """Shell-09, the 2026-09-07 audit: "Good (A)" shares its gate with "Bad
    (R)" and "Skip (S)" beside it -- the surrounding comment says so -- but
    drew with no ``reason``, so it greyed out with no explanation while its
    two neighbours, disabled for the identical cause, said why.
    """
    ctx = SimpleNamespace(textures=None)
    # No rows at all: ``current_label`` answers None, which is the one gate
    # all three buttons share.
    state = SimpleNamespace(labels=review_mode.LabelPass(stage="mesh", rows=[]))

    controls = {c.text: c for c in _draw(ui, ctx, state) if c.kind == "button"}
    good, bad, skip = controls["Good (A)"], controls["Bad (R)"], controls["Skip (S)"]

    assert not good.enabled and not bad.enabled and not skip.enabled
    assert bad.reason == "There is nothing left to label in this pass."
    assert good.reason == bad.reason == skip.reason


def test_a_finding_can_open_its_supporting_examples(ui, svc, monkeypatch):
    """A ranked vector states a conclusion drawn from specific jobs, but until
    now there was no way from the pane to see which ones -- "Apply to forms"
    reuses the vector, and nothing reused the sample.

    "Show examples" has to open onto exactly the jobs the finding was drawn
    from, not merely the library at large: those rows are graded sweep units,
    which ``Filters.matches`` hides from the workshop by design (W3.5's whole
    reason for existing), so landing anywhere but scoped to this set would
    silently show nothing or the wrong thing.
    """
    from warlock.studio import widgets
    from warlock.studio.state import AppState

    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)

    bench = svc.config.bench_dir
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "findings.json").write_text(
        json.dumps(
            {
                "version": 4,
                "params": {},
                "vectors": [
                    {
                        "key": "abc123",
                        "vector": {"lora_weight": 0.9},
                        "n": 8,
                        "accepts": 6,
                        "accept_rate": 0.75,
                        "wilson_low": 0.5,
                        "jobs": ["job-a", "job-b", "job-c"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    ctx = SimpleNamespace(svc=svc, state=state, toast=lambda *a, **kw: None)

    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((900.0, 700.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    review_panes.ReviewPanes()._review_findings(ctx)
    ui.end()
    ui.end_frame()

    buttons = {c.text: c for c in probe.FRAME_CONTROLS if c.kind == "button"}
    assert "Apply to forms" in buttons
    assert "Show examples" in buttons
    assert buttons["Show examples"].enabled

    review_panes.open_examples(ctx, ["job-a", "job-b", "job-c"])

    assert state.mode == "library"
    assert state.library_scroll_to == "job-a"
    assert state.filters.job_ids == frozenset({"job-a", "job-b", "job-c"})


def test_an_open_contrast_offers_a_plan_this_sweep_button(ui, svc, monkeypatch):
    """Review knows which contrasts are unsettled (the axis-verdict lines)
    but never how to settle one -- "Plan this sweep" is the one-click route
    from ``review_mode.suggest_sweeps`` to a filled New-sweep form."""
    from warlock.studio import widgets
    from warlock.studio.state import AppState

    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)

    comparisons = {
        "trellis_gss": [{
            "a": "3.0", "b": "unset", "pairs": 3,
            "a_wins": 2, "b_wins": 1, "ties": 0,
            "sweeps": 1, "prompts": 1, "deltas": {},
        }],
    }
    bench = svc.config.bench_dir
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "findings.json").write_text(
        json.dumps(
            {
                "version": 5,
                "params": {},
                "vectors": [],
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )

    state = AppState()
    ctx = SimpleNamespace(svc=svc, state=state, toast=lambda *a, **kw: None)

    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((900.0, 700.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    review_panes.ReviewPanes()._review_findings(ctx)
    ui.end()
    ui.end_frame()

    buttons = {c.text: c for c in probe.FRAME_CONTROLS if c.kind == "button"}
    assert "Plan this sweep" in buttons

    # Drawing the panel already reached ``review_mode.ensure(ctx)`` to draw the
    # button, so the same ``ReviewState`` the button's own handler would use
    # is already sitting on ``state.review`` -- what the button does is one
    # call away from here, not a second imgui frame with a simulated click.
    suggestion = review_mode.suggest_sweeps({"comparisons": comparisons})[0]
    review_mode.plan_suggestion(state.review, suggestion)
    assert state.review.form.axes[0]["param"] == "trellis_gss"
