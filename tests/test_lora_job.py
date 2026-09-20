"""``_q_lora.LoraOps._lora_train`` must refuse a trainer that reports failure.

The 2026-09-18 audit, finding service-01: the stage used to decide success
from ``weights.exists()`` alone and never read ``result["ok"]``, so a
trainer run that reported ``{"ok": False, "error": ...}`` (the pipelines-01
exit-0 contract every sibling worker follows) still registered as a usable
style whenever partial adapter bytes had already landed on disk.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner import _q_lora as q
from realmspinner import config as config_mod
from realmspinner import fetch, generation, models
from realmspinner.pipelines import blender_run, lora_train


class _FakeProgress:
    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FakeStore:
    async def set_params(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("set_params must not be reached when the trainer failed")


class _FakeTrellis:
    def stop(self) -> None:
        pass


class _FakeWorker:
    def __init__(self, job_dir: Path) -> None:
        self.config = SimpleNamespace(job_dir=lambda job_id: job_dir)
        self.progress = _FakeProgress()
        self.trellis = _FakeTrellis()
        self.store = _FakeStore()
        self._cancel = None

    async def _evict_t2i(self) -> None:
        pass

    def _note_blender(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_lora_train_refuses_when_the_trainer_reports_failure_even_if_weights_partially_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_dir = tmp_path / "job"
    train_dir = job_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "a.png").write_bytes(b"\x89PNG")

    out_dir = job_dir / "lora"
    out_dir.mkdir(parents=True)
    # Partial adapter bytes landed on disk, exactly as the finding describes.
    (out_dir / lora_train.WEIGHTS_NAME).write_bytes(b"partial")

    base_key = next(iter(models.BASE_MODELS))
    monkeypatch.setattr(config_mod, "DEFAULT_BASE_MODEL", base_key)
    monkeypatch.setattr(fetch, "base_model_dir", lambda cfg, spec: tmp_path / "base")
    monkeypatch.setattr(
        lora_train,
        "train_spec",
        lambda *a, **k: SimpleNamespace(),
    )
    weights_path = str(out_dir / lora_train.WEIGHTS_NAME)
    monkeypatch.setattr(
        blender_run,
        "run_worker",
        lambda *a, **k: {
            "ok": False,
            "error": "CUDA out of memory",
            "weights": weights_path,
        },
    )

    def _fail_import_lora(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("import_lora must not be reached when the trainer failed")

    monkeypatch.setattr(generation, "import_lora", _fail_import_lora)

    worker = _FakeWorker(job_dir)
    job = {"id": "job1", "params": {}}

    with pytest.raises(RuntimeError) as excinfo:
        await q.LoraOps._lora_train(worker, job)

    assert "CUDA out of memory" in str(excinfo.value)
