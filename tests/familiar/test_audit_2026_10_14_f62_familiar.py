"""Regression tests for the 2026-10-14 audit, fixer F62, Familiar's test gates.

Findings familiar-core-01 (the fixed hit-rate question set named sections the
Manual renumbering had moved) and familiar-core-09 (the import pin skipped every
relative import).
"""

from __future__ import annotations

import json
from pathlib import Path

from realmspinner.familiar import retrieval

from . import test_familiar_imports as imports_pin

_QUESTIONS = Path(__file__).parent / "data" / "manual_questions.json"

#: The floor the gpu hit-rate test refuses to run under
#: (``test_manual_hitrate_gpu._MIN_QUESTIONS``), restated here so the CPU lane
#: fails on the same number without importing a gpu-marked module.
_MIN_SURVIVORS = 60


def test_the_fixed_manual_questions_still_name_sections_that_exist_at_least_sixty():
    """The 2026-10-14 audit, finding familiar-core-01. Every question is keyed
    on a ``(chapter, title_path)``, and ``title_path`` begins with the chapter's
    *number*. The Manual renumbering rewrote the chapter keys and not that
    prefix, so 25 of 71 questions still named a real section and the gpu test's
    own survivor guard (60) failed -- on a lane that skips without the retrieval
    row, so nothing on the CPU lane ever said so. This counts survivors with
    the same lookup the gpu test uses and needs no embedder."""
    sections = {(c.chapter, c.title_path) for c in retrieval.Index.build().chunks}
    questions = json.loads(_QUESTIONS.read_text(encoding="utf-8"))
    alive = [q for q in questions if (q["chapter"], q["section"]) in sections]
    assert len(alive) >= _MIN_SURVIVORS, (
        f"only {len(alive)} of {len(questions)} fixed questions name a section that exists: "
        "refresh the leading chapter number of tests/familiar/data/manual_questions.json"
    )


def test_the_hitrate_gpu_test_and_this_gate_agree_on_the_survivor_floor():
    """The floor above is a copy, so it is read back out of the gpu module's
    source rather than trusted: a changed floor there fails here."""
    source = (Path(__file__).parent / "test_manual_hitrate_gpu.py").read_text(encoding="utf-8")
    assert f"_MIN_QUESTIONS = {_MIN_SURVIVORS}" in source


def _plant(tmp_path: Path, relative: str, body: str) -> tuple[Path, Path]:
    """A throwaway package directory standing in for ``familiar/``."""
    package = tmp_path / "familiar"
    target = package / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return package, target


def test_the_familiar_import_pin_resolves_relative_imports_that_leave_the_package(tmp_path):
    """The 2026-10-14 audit, finding familiar-core-09. ``from ..service import x``
    climbs out of ``familiar/`` into a peer layer; the pin used to ``continue``
    on every relative import and never looked."""
    for body, expected in (
        ("from .. import service\n", "realmspinner.service"),
        ("from ..service import door\n", "realmspinner.service"),
        ("from .. import queue\n", "realmspinner.queue"),
        ("from ..queue import Worker\n", "realmspinner.queue"),
        ("from realmspinner import service\n", "realmspinner.service"),
    ):
        package, target = _plant(tmp_path, "planted.py", body)
        problems = imports_pin._violations(target, package)
        assert any(expected in p for p in problems), (body, problems)

    # one level deeper: familiar/sub/planted.py, level 3 reaches the same place
    package, target = _plant(tmp_path, "sub/planted.py", "from ... import service\n")
    assert imports_pin._violations(target, package)

    # and the ordinary sibling import stays clean
    package, target = _plant(tmp_path, "ok.py", "from . import router\nfrom .router import x\n")
    assert imports_pin._violations(target, package) == []
