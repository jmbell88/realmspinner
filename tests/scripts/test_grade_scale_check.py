"""``scripts/grade_scale_check.py`` against a seeded store.

Loaded the way ``tests/test_make_packs.py`` loads ``make_packs.py`` --
``importlib`` off the file path, scripts/ pushed onto ``sys.path`` only for
the duration of the import so it does not leak into other tests' module
namespace. The report itself is exercised end to end (``graded_rows`` against
a real ``JobStore``, through ``report``'s stdout) rather than against
hand-built rows, because the join between ``store.latest_verdicts()`` and the
prediction/revisit logic is exactly what a reader script can get wrong
silently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from warlock.service import verdicts as svc_verdicts

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def check():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "grade_scale_check", SCRIPTS / "grade_scale_check.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["grade_scale_check"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


# Ten grades on one shared vector (``platform: pc``), chosen so every fact the
# report states is deterministic and known ahead of time:
#
# - not every grade is +-3 (prediction 1's failure condition does not fire);
# - the mass off +-3 leans negative -- 6 rows below zero against 3 above
#   (left-heavy, prediction 1 holds);
# - grade 0 is used exactly once (prediction 3 holds);
# - the bucket has n=10, clearing the revisit condition's n>=10 bar, and its
#   grades are not all equal, so its standard deviation is above zero.
_GRADES = [-5, -4, -3, -3, -2, -1, 0, 1, 2, 3]

# Tagged so ``sharp-detail`` (4) outranks ``holes`` (3), which in turn
# outranks the two one-off distractors -- prediction 2's target pair, by a
# clear margin so no tie-break in ``Counter.most_common`` is in play.
_REASONS = [
    ("holes", "broken"),
    ("holes",),
    ("sharp-detail",),
    ("holes",),
    ("sharp-detail",),
    ("sharp-detail",),
    (),
    ("sharp-detail",),
    ("bad-shape",),
    (),
]


def _seed(svc) -> None:
    for grade, reasons in zip(_GRADES, _REASONS, strict=True):
        job_id = svc.store.create(
            "image", "a chest", {"platform": "pc"}, stage="model", status="done"
        )
        svc_verdicts.record_verdict(svc, job_id, grade=grade, reasons=reasons, source="human")


def test_the_check_reports_each_prediction_against_a_seeded_store(check, svc, capsys):
    _seed(svc)

    rows = check.graded_rows(svc.store, None)
    assert len(rows) == 10

    check.report(rows)
    out = capsys.readouterr().out

    assert "latest human model verdicts: 10" in out
    # Prediction 1: not degenerate (rest > 0) and left-heavy (6 negative vs 3
    # positive off the +-3 spikes... the full corpus, actually: -5,-4,-3,-3,-2,-1
    # below zero (6) against 1,2,3 above zero (3)).
    assert (
        "1. HOLDS -- 7/10 row(s) grade outside +-3, and the mass is left-heavy"
        " (6 negative vs 3 positive)"
    ) in out
    assert "2. HOLDS -- top tags are sharp-detail, holes" in out
    assert "3. HOLDS -- grade 0 filed 1 time(s)" in out
    assert "any bucket reaches n >= 10: yes" in out
    assert "largest bucket with grade sd > 0:" in out
    assert "(n=10)" in out


def test_the_tag_scope_narrows_to_one_corpus(check, svc):
    """``--tag`` (``graded_rows``'s ``tag`` argument) scopes to jobs carrying
    that corpus tag on the job row -- reusing ``hole_audit_vs_grade.tagged_jobs``
    rather than a second reading of the tags column."""
    tagged_job = svc.store.create(
        "image", "a chest", {"platform": "pc"}, stage="model", status="done"
    )
    svc.store.set_meta(tagged_job, tags="props-v1")
    svc_verdicts.record_verdict(svc, tagged_job, grade=3, source="human")
    other_job = svc.store.create(
        "image", "a lamp", {"platform": "pc"}, stage="model", status="done"
    )
    svc_verdicts.record_verdict(svc, other_job, grade=-3, source="human")

    scoped = check.graded_rows(svc.store, "props-v1")

    assert [r["job_id"] for r in scoped] == [tagged_job]
