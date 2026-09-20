"""``lora_train_worker.main()``'s failure contract.

Every sibling worker that stages a JSON result (``fetch_worker``,
``pack_worker``) wraps its real work in ``except Exception`` and turns a
failure into ``{"ok": False, ...}`` written to ``result_path``. Driven
in-process with a stubbed stdin and a monkeypatched ``train``, so this costs
no weights and no GPU.
"""

from __future__ import annotations

import io
import json

from realmspinner.pipelines import lora_train_worker as worker


def _spec(tmp_path) -> dict:
    return {
        "op": "train",
        "base_dir": str(tmp_path),
        "images": [],
        "out_dir": str(tmp_path / "out"),
        "result_path": str(tmp_path / "result.json"),
        "trigger": "x",
        "steps": 1,
        "rank": 4,
        "learning_rate": 1e-4,
        "resolution": 64,
        "seed": 0,
    }


def test_lora_train_worker_main_reports_a_train_failure_as_a_staged_result_not_an_uncaught_traceback(  # noqa: E501
    tmp_path, monkeypatch
):
    """2026-09-18 audit (pipelines-01): a CUDA OOM raised mid-training used to
    propagate straight out of ``main()`` as an uncaught traceback with no
    result file at all, unlike every sibling worker.
    """
    spec = _spec(tmp_path)
    result_path = spec["result_path"]

    def _boom(_spec):
        raise RuntimeError("CUDA out of memory. (a realistic OOM message)")

    monkeypatch.setattr(worker, "train", _boom)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(spec)))

    code = worker.main()

    assert code == 0
    from pathlib import Path

    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "CUDA out of memory" in result["error"]


def test_lora_train_worker_main_stages_a_successful_result(tmp_path, monkeypatch):
    spec = _spec(tmp_path)

    def _fake_train(_spec):
        return {"ok": True, "steps": 1, "images": 0, "rank": 4, "loss": 0.1, "weights": "x"}

    monkeypatch.setattr(worker, "train", _fake_train)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(spec)))

    code = worker.main()

    assert code == 0
    from pathlib import Path

    result = json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))
    assert result["ok"] is True
