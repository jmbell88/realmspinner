"""Argument refusals in ``Text2Image._generate`` that must fire before any
model is touched.

No GPU, no diffusers: these are pure argument checks, so the fixture builds a
``Text2Image`` the way ``test_conditioning.py``'s does and calls the private
method directly. If a check here ever needed ``self._pipe`` or ``self.load``,
that would be a sign it moved to the wrong place in ``_generate``.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from PIL import Image

from realmspinner import models
from realmspinner.pipelines import text2image
from realmspinner.pipelines.conditioning import Conditioning


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


def test_cancel_set_during_conditioning_attach_stops_before_sampling(t2i, tmp_path):
    """pipelines-02 (2026-09-08 audit). ``_conditioned`` checked
    ``cancel_event`` nowhere, so a cancel pressed while the ~1.2 GB IP-Adapter
    was loading from disk went unobserved until the per-step diffusion
    callback fired -- several seconds and a VRAM allocation later. The check
    now runs right after that load, through the same teardown path every
    other failure in this function already takes.

    A stubbed ``self._pipe`` (``MagicMock``, no real diffusers class needed
    for the IP-Adapter-only branch) rather than ``self.load()``, matching this
    module's no-GPU, no-diffusers rule.
    """
    adapter = models.IP_ADAPTERS["plus"]
    weights = tmp_path / adapter.dir_name / adapter.subfolder / adapter.weight_name
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8)).save(ref)

    t2i._pipe = MagicMock()
    cond = Conditioning(ip_adapter="plus", ip_image=ref)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(text2image.JobCancelled):
        t2i._conditioned(cond, cancel_event)

    # The load already happened -- the check fires *after* the sub-load
    # rather than skipping it, so a cancel observed here still paid the
    # load's own cost and no more than that.
    t2i._pipe.load_ip_adapter.assert_called_once()
    # And teardown ran, undoing exactly what had already attached.
    t2i._pipe.unload_ip_adapter.assert_called_once()
