"""Regression test for the 2026-09-23 (second run) audit's poser-05: an
op-level runtime error used to reach ``blender_worker.main``'s caller as a
raw traceback instead of the "sentence and an exit code" contract the
function's own docstring states for a malformed spec.

``blender_worker.py`` is the only module that imports ``bpy`` (CLAUDE.md); no
real Blender is exercised here, matching the existing stdin/result contract
tests in ``tests/pipelines/test_rig_worker.py`` (``blender_worker.OPS`` is
monkeypatched instead).
"""

from __future__ import annotations

import io
import json
import sys
import types


def test_an_op_level_runtime_error_is_reported_as_a_sentence_and_an_exit_code_never_as_a_traceback(
    monkeypatch, capsys
):
    from realmspinner.pipelines import blender_worker

    def explode(_bpy, _spec):
        # A routine refusal an op raises ordinarily -- e.g. op_rig's own
        # "no mesh to rig at ..." -- not a bug in the worker itself.
        raise ValueError("no mesh to rig at the given path")

    monkeypatch.setitem(blender_worker.OPS, "rig", explode)
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"op": "rig", "result_path": "unused.json"}))
    )
    code = blender_worker.main()
    err = capsys.readouterr().err
    # A distinct, non-zero code: 0 is success, 2 is a bad spec/unknown op, 3
    # is no bpy -- none of those already mean "the op itself raised".
    assert code not in (0, 2, 3)
    assert code != 0
    # The sentence, not a traceback: no frame markers, and the exception's
    # own message reaches stderr for the job's own error text to quote.
    assert "no mesh to rig at the given path" in err
    assert "Traceback" not in err
