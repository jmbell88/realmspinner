"""``Text2Image._generate``'s declarative tile refusals run before any model
is touched.

The 2026-09-14 audit (pipelines-07): the tile-requires-SDXL-family refusal ran
*after* ``self.load(on_state)`` had already paid for a full checkpoint load,
even though the check depends on nothing ``load`` produces (``self.spec`` is
set in ``__init__``). Its sibling, the tile+sheet/tilesheet mutual-exclusion
refusal, was already ahead of ``load`` -- ``tests/pipelines/test_text2image.py`` pins
that one. This file pins the one that moved.

No GPU, no diffusers: this is a pure argument check, so the fixture builds a
``Text2Image`` the way ``test_text2image.py``'s does, against a non-SDXL
(FLUX2-klein) base model, and spies on ``load`` rather than calling it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from realmspinner import models
from realmspinner.pipelines import text2image


@pytest.fixture
def flux_t2i(tmp_path, monkeypatch):
    t2i = text2image.Text2Image(models.BASE_MODELS["flux_klein"], tmp_path)
    monkeypatch.setattr(t2i, "load", MagicMock(side_effect=AssertionError("load() was called")))
    return t2i


def test_tile_on_a_non_sdxl_family_is_refused_before_the_checkpoint_loads(flux_t2i, tmp_path):
    with pytest.raises(RuntimeError, match="not an SDXL-family checkpoint"):
        flux_t2i._generate(
            "a prompt",
            tmp_path / "out.png",
            tile=True,
        )
    flux_t2i.load.assert_not_called()
