"""Regressions for the 2026-09-16 audit, findings docs-02 and docs-03 (the
`fix-docs` brief's second and third records; the first, `_LEAD`'s
abbreviation truncation, is pinned in ``tests/test_changelog.py`` where the
function it covers already has a test module).

* docs-02 -- ``service/validation.py``'s ``DERIVED_PARAMS`` comment cited
  ``tests/test_jobs_resubmit.py::test_every_door_that_copies_params_rerolls_the_seeds``,
  a file that does not exist anywhere in the tree; the real test lives in
  ``tests/test_rerun_regressions.py``.
* docs-03 -- ``docs/measurements/2026-08-18-tile-sheet-grid.md`` still
  presented the tile-variety question as open among three unchosen
  candidates, after ``TODO.md``'s P10 record shows candidate 1 was chosen
  and shipped 2026-08-29 as the separate "Materials and Terrain set" job
  kind, with this document's own grid mechanism relabelled "Grid (legacy)"
  and its verdict moved to ``TODO.md``'s P15.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_derived_params_comment_cites_the_real_test_file():
    """The ``DERIVED_PARAMS`` comment in ``service/validation.py`` must point
    at a test file that actually exists, and must not point at the deleted
    (never-existed) ``tests/test_jobs_resubmit.py``.
    """
    assert not (ROOT / "tests" / "test_jobs_resubmit.py").exists(), (
        "tests/test_jobs_resubmit.py exists now -- re-check docs-02 against "
        "the current tree"
    )
    assert (ROOT / "tests" / "test_rerun_regressions.py").exists()

    validation = (ROOT / "src" / "warlock" / "service" / "validation.py").read_text(
        encoding="utf-8"
    )
    assert "tests/test_jobs_resubmit.py" not in validation, (
        "service/validation.py's DERIVED_PARAMS comment still cites "
        "tests/test_jobs_resubmit.py, which does not exist in this tree"
    )
    assert (
        "tests/test_rerun_regressions.py"
        "::test_every_door_that_copies_params_rerolls_the_seeds" in validation
    ), (
        "service/validation.py's DERIVED_PARAMS comment must cite "
        "tests/test_rerun_regressions.py::"
        "test_every_door_that_copies_params_rerolls_the_seeds, where that "
        "test actually lives"
    )


def test_tile_sheet_grid_measurement_notes_the_materials_and_terrain_supersession():
    """``docs/measurements/2026-08-18-tile-sheet-grid.md`` must record that
    ``TODO.md``'s P10 resolved the feature-level question on 2026-08-29 by
    shipping "Materials and Terrain set" (candidate 1) as the primary path,
    that the grid mechanism this document covers now ships only as "Grid
    (legacy)", and must still point at TODO.md's P15 for the new path's own
    verdict -- rather than presenting the tile-variety problem as an entirely
    open decision among three candidates.
    """
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    assert "P10's tile-sheet half" in todo and "Materials and Terrain set" in todo, (
        "TODO.md no longer records P10's tile-sheet resolution the way this "
        "test expects -- re-check docs-03 against the current tree"
    )

    doc = (
        ROOT / "docs" / "measurements" / "2026-08-18-tile-sheet-grid.md"
    ).read_text(encoding="utf-8")
    assert "Materials and Terrain set" in doc, (
        "docs/measurements/2026-08-18-tile-sheet-grid.md does not mention "
        "'Materials and Terrain set', the job kind TODO.md's P10 shipped "
        "2026-08-29 to resolve the question this document leaves open"
    )
    assert "Grid (legacy)" in doc, (
        "docs/measurements/2026-08-18-tile-sheet-grid.md does not say the "
        "grid mechanism it measures now ships only as 'Grid (legacy)'"
    )
    assert "P15" in doc, (
        "docs/measurements/2026-08-18-tile-sheet-grid.md does not point to "
        "TODO.md's P15 for the new path's verdict"
    )
