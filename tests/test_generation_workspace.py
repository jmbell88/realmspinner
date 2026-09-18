"""The Create plan is a view of the existing request, not a second planner."""

from __future__ import annotations

import pytest

from warlock.studio.modes.create.engine import assets as create_assets
from warlock.studio.modes.create.engine import recipe as create_recipe
from warlock.studio.modes.create.ui import workspace as generation_workspace
from warlock.studio.state import default_form_2d


@pytest.mark.parametrize(
    ("asset_type", "minimum_generations"),
    (
        ("image", 1),
        ("model_3d", 1),
        ("seamless_material", 1),
        ("tileset", 1),
        # A sheet always includes its character reference plus its sheet work.
        ("sprite_sheet", 2),
        # And a character includes *none*: it draws no images at all. Zero is
        # the honest floor here rather than a hole in the table -- the plan
        # still has a duration and still has stages, which is what this test is
        # about.
        ("character", 0),
    ),
)
def test_generation_plan_covers_every_create_outcome(asset_type, minimum_generations):
    form = default_form_2d()
    form["asset_type"] = asset_type
    form["generation_type"] = asset_type
    create_assets.sync_legacy_fields(form)

    plan = generation_workspace.plan_for(form)

    assert plan.candidates >= 1
    assert plan.generations >= minimum_generations
    assert plan.duration
    assert plan.stages


def test_sprite_plan_states_the_compound_work_plainly():
    form = default_form_2d()
    form["asset_type"] = form["generation_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)

    plan = generation_workspace.plan_for(form)

    assert "character reference" in plan.stages
    assert "sheet generation" in plan.stages


# --- nothing in the tray is drawn where it cannot be pressed -----------------


def test_the_tray_shows_one_whole_row_rather_than_two_half_rows():
    """The tray is a fixed-height strip. Six results filled its three columns
    twice over, and the second row's cards were drawn with their actions below
    the fold, where nothing can press them.

    ``/exercise-mode create`` reported fifteen clipped controls, every one of
    them a result-card action. No test could: a clipped button is still drawn,
    and the smoke suite only asks whether a pane builds.
    """
    import inspect

    from warlock.studio.modes.create.ui import workspace as gw

    assert gw._RESULT_COLUMNS == 3
    source = inspect.getsource(gw._recent_results)
    assert "_RESULT_COLUMNS" in source, "the cap and the grid width are one fact"
    # And the grid is built from the same number, so they cannot drift.
    assert "_RESULT_COLUMNS" in inspect.getsource(gw._result_grid)


def test_the_result_actions_are_two_per_row():
    """Four full-width buttons under a 72 dp thumbnail make a card taller than
    the strip that holds it."""
    import inspect

    from warlock.studio.modes.create.ui import workspace as gw

    source = inspect.getsource(gw._result_card)
    assert "_half_width()" in source
    assert "(-1, 0)" not in source, "a full-width action is a row of its own"
    assert source.count("imgui.same_line()") >= 2


def test_make_3d_always_pairs_with_a_neighbour_on_a_candidate_card():
    """The 2026-09-07 audit, finding create-08: on a candidate card (``group``
    is not None) "Make 3D" used to follow Rerun with ``same_line()`` fired
    only ``if group is None`` -- so on the one card shape where a "Keep"
    button also exists, nothing joined Make 3D to the row above it and it sat
    alone at half width with its other half blank, against this very
    function's own "two per row" design one comment up. Rerun and Make 3D are
    joined unconditionally now; the button left without a same-row partner
    when five actions cannot divide evenly by two is Keep instead.
    """
    import inspect

    from warlock.studio.modes.create.ui import workspace as gw

    source = inspect.getsource(gw._result_card)

    # The bug, named literally: the join fired only in the absence of Keep.
    assert "if group is None:\n        imgui.same_line()" not in source

    # Keep no longer claims a same_line(): nothing between its button call and
    # the Rerun comment block joins it to what follows.
    before_rerun, _, _ = source.partition("# **Rerun is live on a failure.**")
    keep_onward = before_rerun[before_rerun.index("Keep##result-keep") :]
    assert "imgui.same_line()" not in keep_onward

    # Rerun and Make 3D are joined unconditionally, in that order.
    after_rerun = source[source.index('"Rerun##result-rerun-{job_id}"') :]
    same_line_pos = after_rerun.index("imgui.same_line()")
    make3d_pos = after_rerun.index('"Make 3D##result-3d-{job_id}"')
    assert same_line_pos < make3d_pos


def test_the_candidate_grid_scrolls_rather_than_truncating():
    """A count of 8 means eight candidates and choosing between them is the
    whole purpose, so this grid cannot be trimmed to a row the way the results
    grid is."""
    import inspect

    from warlock.studio.modes.create.ui import workspace as gw

    source = inspect.getsource(gw._candidate_grid)
    assert 'begin_child("generation-candidate-scroll"' in source
    assert "end_child()" in source


def test_the_keeper_pill_is_not_the_ranker():
    """The card may not label ``rank.score`` as the trained probe.

    ``params["rank"]`` is written by ``_q_mesh._rank_reference`` from
    ``pipelines.rank.score``; ``service.judge`` is called only from Review and
    never persists its probability onto the row, so this module has no probe
    answer to draw. Drawing the ranker under the judge's name asserted a
    keep-probability nobody computed -- and, before the speckle floor was fixed,
    asserted it was **zero** on every ordinary generation.

    A source test because the claim is about a string the frame draws, and this
    module's card is drawn straight into imgui.
    """
    import inspect

    src = inspect.getsource(generation_workspace)
    body = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "likely a keeper" not in body
    assert "judge" not in body.lower()
    assert 'f"rank {' in body


def _model_form(prompt):
    from warlock.studio.modes.create.ui import settings_2d

    form = default_form_2d()
    form["asset_type"] = form["generation_type"] = "3d_model"
    create_assets.sync_legacy_fields(form)
    form["prompt"] = prompt
    return settings_2d, form


def test_open_form_prompt_is_advisory_and_does_not_block_generate():
    """The lint may not join ``problems_for``.

    Every member of that list disables Generate, and an audit-flagged open form
    still grades usable two times in five
    (dev/measurements/2026-09-02-fantasy-v1.md) -- so blocking would be the app
    asserting a certainty the corpus does not support. Fails against the
    unfixed code, where ``advisories_for`` does not exist.
    """
    settings_2d, form = _model_form("a wooden cart wheel with spokes")

    advisories = create_recipe.advisories_for(None, form)

    assert len(advisories) == 1
    assert advisories[0].field == "prompt"
    assert "2 times in 5" in str(advisories[0])
    # Never a verdict.
    assert "will fail" not in str(advisories[0]).lower()
    # And it is a different type from the thing that stops a press.
    from warlock.studio import problems

    assert isinstance(advisories[0], problems.Advisory)
    assert not isinstance(advisories[0], problems.Problem)


def test_a_closed_subject_draws_no_advisory():
    settings_2d, form = _model_form("a solid stone barrel, banded with iron")
    assert create_recipe.advisories_for(None, form) == []


def test_the_lint_is_only_about_the_reconstruction_arm():
    """A picture of a birdcage is a fine picture; only a mesh has a back."""
    settings_2d, form = _model_form("an ornate birdcage")
    assert create_recipe.advisories_for(None, form)
    form["asset_type"] = form["generation_type"] = "image"
    create_assets.sync_legacy_fields(form)
    assert create_recipe.advisories_for(None, form) == []


def test_the_words_are_matched_whole_and_named_back():
    from warlock.studio.modes.create.engine import recipe as create_recipe

    assert create_recipe.open_form_words("a cart wheel with spokes") == ("wheel", "spokes")
    # "netting" is a word in the list; "vignetting" is not this word.
    assert create_recipe.open_form_words("heavy vignetting") == ()
    assert create_recipe.open_form_words("a barrel") == ()
