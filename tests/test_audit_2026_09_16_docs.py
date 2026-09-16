"""Regressions for the 2026-09-16 audit, finding docs-02 (the `fix-docs`
brief's second record; the first, `_LEAD`'s abbreviation truncation, is
pinned in ``tests/test_changelog.py`` where the function it covers already
has a test module). docs-03, which checked ``docs/measurements/`` and
``TODO.md`` -- both moved to ``dev/`` on 2026-09-16 -- lives in
``dev/tests/test_audit_2026_09_16_docs.py``.

* docs-02 -- ``service/validation.py``'s ``DERIVED_PARAMS`` comment cited
  ``tests/test_jobs_resubmit.py::test_every_door_that_copies_params_rerolls_the_seeds``,
  a file that does not exist anywhere in the tree; the real test lives in
  ``tests/test_rerun_regressions.py``.
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
