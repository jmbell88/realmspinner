"""``scripts/agent_bench.py --show`` against a fixture transcript.

Loaded the way ``tests/scripts/test_grade_scale_check.py`` loads
``grade_scale_check.py`` -- ``importlib`` off the file path, ``scripts/``
pushed onto ``sys.path`` only for the duration of the import, so it does not
leak into other tests' module namespace.

Only ``--show`` is exercised here. ``--serve`` boots the real windowed app
with a real GL context and hands off to ``studio.main.run()`` -- exactly the
thing this suite's own lane rules forbid (no window, no GL, ``--dist
loadfile`` keeps at most one imgui context alive per process) and exactly
the thing ``scripts/exercise_mode.py``'s own docstring says a driver like
this "must not run while pytest is running". There is no test for it here,
and there should not be one: proving it boots is a manual, once-per-change
check a human runs by hand, the same way ``exercise_mode.py`` and
``screenshot_modes.py`` are.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
CHAIR = ROOT / "tests" / "fixtures" / "agent_transcripts" / "chair.jsonl"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("agent_bench", SCRIPTS / "agent_bench.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_bench"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_show_prints_the_hand_authored_chair_transcript_without_raising(bench, capsys) -> None:
    exit_code = bench._show(CHAIR)
    out = capsys.readouterr().out

    assert exit_code == 0
    # Every one of chair.jsonl's seven lines is a success (see
    # tests/fixtures/agent_transcripts/chair.jsonl): six primitives and one
    # closing clay_material call over all six -- 6 * 1 + 6 = 12 uids made.
    assert "clay_add_primitive" in out
    assert "clay_material" in out
    assert "7 call(s), 7 ok, 0 refused, 12 uid(s) produced" in out


def test_show_marks_a_refused_call_and_still_reports_a_clean_count(
    bench, tmp_path, capsys
) -> None:
    transcript = tmp_path / "mixed.jsonl"
    transcript.write_text(
        '{"tool": "clay_add_primitive", "arguments": {"generator": "box"}, '
        '"ok": true, "made": [1]}\n'
        '{"tool": "clay_nonexistent", "arguments": {}, "ok": false, "made": []}\n',
        encoding="utf-8",
    )

    exit_code = bench._show(transcript)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "REFUSED" in out
    assert "2 call(s), 1 ok, 1 refused, 1 uid(s) produced" in out
