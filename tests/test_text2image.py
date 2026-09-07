"""Argument refusals in ``Text2Image._generate`` that must fire before any
model is touched.

No GPU, no diffusers: these are pure argument checks, so the fixture builds a
``Text2Image`` the way ``test_conditioning.py``'s does and calls the private
method directly. If a check here ever needed ``self._pipe`` or ``self.load``,
that would be a sign it moved to the wrong place in ``_generate``.
"""

from __future__ import annotations

import pytest

from warlock import models
from warlock.pipelines import text2image


@pytest.fixture
def t2i(tmp_path):
    return text2image.Text2Image(models.BASE_MODELS["sdxl_cfg"], tmp_path)


def test_tile_combined_with_sheet_is_refused_before_loading(t2i, tmp_path):
    """pipelines-02 (2026-09-07 audit): circular padding wraps the whole
    frame, and a contact sheet's cells are different subjects/tiles side by
    side -- wrapping its outer edge would bleed the first cell into the last.
    Only caller discipline enforced the exclusion before this check existed.
    """
    with pytest.raises(RuntimeError, match="tile cannot be combined"):
        t2i._generate(
            "a prompt",
            tmp_path / "out.png",
            tile=True,
            sheet=True,
        )


def test_tile_combined_with_tilesheet_is_refused_before_loading(t2i, tmp_path):
    """Same defect, the other flag: ``tilesheet`` shares ``sheet``'s no-wrap
    rule for the same reason spelled a different way (its cells are sixty-four
    different tiles, not one repeating surface)."""
    with pytest.raises(RuntimeError, match="tile cannot be combined"):
        t2i._generate(
            "a prompt",
            tmp_path / "out.png",
            tile=True,
            tilesheet=True,
        )
