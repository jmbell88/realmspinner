"""The 2026-10-14 audit's service-queue-07 remainder: the matting client holds the model lease."""

from __future__ import annotations

import numpy as np
from PIL import Image

from realmspinner import leases
from realmspinner.pipelines import matting


def test_the_matting_child_runs_under_the_model_lease_and_unload_does_not(monkeypatch, tmp_path):
    """The model runs in a child no in-process lease saw, so a Download could
    start while it was mid-``from_pretrained``. ``unload`` runs on the loop
    thread and must not park behind a maintainer."""
    seen: dict[str, int] = {}

    def fake_request(payload, key):
        seen["request"] = leases.MODELS.users
        Image.fromarray(np.full((4, 4), 255, np.uint8)).save(payload["output"])

    monkeypatch.setattr(matting, "_request", fake_request)
    monkeypatch.setattr(matting, "_cache", {})

    found = matting._model_mask(Image.new("RGB", (4, 4)), tmp_path, "cpu")

    assert seen == {"request": 1}
    assert bool(found.all())
    assert leases.MODELS.users == 0

    monkeypatch.setattr(
        matting, "_stop_child", lambda: seen.setdefault("unload", leases.MODELS.users)
    )
    matting.unload()
    assert seen["unload"] == 0
