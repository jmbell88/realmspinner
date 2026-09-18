"""The candidate score: what the reference report already knew, as a number.

Advisory by construction -- nothing rejects a candidate. Its whole job is to
put the most likely one first in a strip of eight.
"""

from __future__ import annotations

from warlock.pipelines import rank, reference


def _report(**kwargs):
    return reference.Report(**kwargs).as_dict()


def test_a_refused_reference_scores_zero():
    assert rank.composition_score(_report(ok=False, reasons=("too small",))) == 0.0


def test_a_clean_reference_at_the_target_occupancy_scores_one():
    assert (
        rank.composition_score(_report(occupancy=reference.DEFAULT_OCCUPANCY, components_major=1))
        == 1.0
    )


def test_missing_the_target_occupancy_costs_something_but_not_everything():
    near = rank.composition_score(_report(occupancy=0.70, components_major=1))
    far = rank.composition_score(_report(occupancy=0.20, components_major=1))
    assert 0.0 < far < near < 1.0


def test_a_second_object_is_penalised():
    one = rank.composition_score(_report(occupancy=0.78, components_major=1))
    two = rank.composition_score(_report(occupancy=0.78, components_major=2))
    assert two < one


def test_running_off_the_edge_is_penalised():
    clean = rank.composition_score(_report(occupancy=0.78, components_major=1))
    cropped = rank.composition_score(
        _report(occupancy=0.78, components_major=1, touches=("left",))
    )
    assert cropped < clean


def test_warnings_cost_less_than_reasons():
    warned = rank.composition_score(
        _report(occupancy=0.78, components_major=1, warnings=("close to the edge",))
    )
    clean = rank.composition_score(_report(occupancy=0.78, components_major=1))
    refused = rank.composition_score(_report(ok=False, reasons=("too small",)))
    assert refused < warned < clean


def test_no_report_at_all_is_a_middling_score_not_a_zero():
    # A job whose measurement failed is unknown, not bad -- scoring it zero
    # would sort it below a candidate that was actually measured and refused.
    assert 0.0 < rank.composition_score(None) < 1.0


def test_composition_score_treats_a_leaked_mask_report_as_unmeasured_not_low():
    """pipelines-03 (2026-09-08 audit). ``reference.unmeasured()`` -- what
    ``measure()`` returns whenever the corner-flood fill leaks through the
    subject, the documented common case of a light subject on a light
    background -- is a *present* report (``ok=True``) carrying a placeholder
    ``occupancy=0.0``. Scoring that placeholder as a real measurement charged
    it the full occupancy and warning cost (~0.59) instead of this module's
    own "unknown, not bad" mid-range score for a report that says nothing.
    """
    leaked = reference.unmeasured("the mask leaked").as_dict()
    assert rank.composition_score(leaked) == rank.UNMEASURED


def test_the_score_is_the_composition_when_there_is_no_anchor():
    out = rank.score(_report(occupancy=0.78, components_major=1))
    assert out["score"] == out["composition"] == 1.0
    assert out["anchor"] is None


def test_an_anchor_cosine_moves_the_score_and_is_recorded():
    report = _report(occupancy=0.78, components_major=1)
    close = rank.score(report, anchor_cosine=0.9)
    far = rank.score(report, anchor_cosine=0.1)
    assert close["score"] > far["score"]
    assert close["anchor"] == 0.9


def test_every_score_stays_inside_zero_and_one():
    for cosine in (-1.0, 0.0, 1.0):
        for report in (None, _report(ok=False), _report(occupancy=0.01, components_major=5)):
            out = rank.score(report, anchor_cosine=cosine)
            assert 0.0 <= out["score"] <= 1.0


def test_speckle_does_not_floor_the_composition_score():
    """The defect this module shipped with, pinned.

    A real reference carries 15-18 connected components and one *subject*
    (dev/measurements/2026-08-17-reference-source-bench.md). Charging
    COMPONENT_COST against the raw count took 2.25 off a base of 1.0, so the
    clamp fired on every image and the 0.6-weighted composition term was a
    constant. Fails against the unfixed code, which returns 0.0 here.
    """
    speckled = _report(occupancy=reference.DEFAULT_OCCUPANCY, components=18, components_major=1)
    assert rank.composition_score(speckled) == 1.0


def test_the_raw_component_count_is_not_what_is_charged_for():
    clean = _report(occupancy=0.78, components=1, components_major=1)
    speckled = _report(occupancy=0.78, components=40, components_major=1)
    assert rank.composition_score(speckled) == rank.composition_score(clean)


def test_a_report_written_before_the_filter_existed_is_unmeasured_not_floored():
    """No ``components_major`` key at all -- an August report read back off disk."""
    old = {"ok": True, "occupancy": reference.DEFAULT_OCCUPANCY, "components": 17}
    assert rank.composition_score(old) == 1.0


def test_a_refused_reference_never_outranks_an_accepted_one_once_anchor_and_preference_are_blended_in():  # noqa: E501
    """2026-09-18 audit (pipelines-02).

    A refused candidate's zero composition used to enter the same blend as
    everyone else's, so a near-perfect anchor/preference term could pull it
    back above an accepted-but-weak candidate (0.55 vs 0.31 -- the reproduction
    in the audit's probe), contradicting ``composition_score``'s own "a
    candidate that is going to be refused at promotion belongs last."
    """
    refused_report = {"ok": False, "reasons": ("too small",), "codes": ("occupancy",)}
    refused = rank.score(refused_report, anchor_cosine=1.0, preference=25.0)

    accepted_report = {
        "ok": True,
        "reasons": (),
        "occupancy": 0.06,
        "components_major": 1,
        "touches": ("left",),
        "warnings": ("The subject touches the edge of the frame.",),
    }
    accepted = rank.score(accepted_report, anchor_cosine=-1.0, preference=None)

    assert refused["score"] <= accepted["score"]
    assert refused["score"] == 0.0
