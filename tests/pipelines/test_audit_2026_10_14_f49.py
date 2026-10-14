"""The 2026-10-14 audit, findings pipelines-image-01/-03/-05/-07 (F49).

No GPU and no diffusers, the same shape as ``test_conditioning.py``: the fixes
are all decisions made before a pipeline is called.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from realmspinner import models
from realmspinner.pipelines import control, text2image
from realmspinner.pipelines.conditioning import Conditioning


@pytest.fixture
def t2i(tmp_path):
    return text2image.Text2Image(models.BASE_MODELS["sdxl_cfg"], tmp_path)


def _frame_after_fit(t2i, width, height, *, size=None):
    cond = Conditioning(init_image=Path("init.png"), strength=0.6)
    extra = {"image": Image.new("RGB", (width, height), (10, 20, 30))}
    t2i._fit_init(cond, extra, size)
    return t2i._init_frame(cond, extra, size), extra


def test_init_frame_is_clamped_to_the_models_working_range_for_oversized_and_tiny_references(
    t2i,
):
    """A 4000x4000 start image made SDXL sample and VAE-decode at 4000x4000 with
    the checkpoint already resident; a 64x64 one sampled at 64x64."""
    frame, extra = _frame_after_fit(t2i, 4000, 4000)
    assert frame == (1024, 1024)
    assert extra["image"].size == (1024, 1024)

    frame, extra = _frame_after_fit(t2i, 4000, 2000)
    assert frame == (1024, 512)
    assert extra["image"].size == frame

    frame, extra = _frame_after_fit(t2i, 64, 64)
    assert frame == (512, 512)
    assert extra["image"].size == (512, 512)

    # Inside the range nothing moves: Inker's 1024x448 crop keeps its aspect.
    frame, _ = _frame_after_fit(t2i, 1024, 448)
    assert frame == (1024, 448)


def test_a_mask_is_fitted_with_its_picture(t2i):
    cond = Conditioning(
        init_image=Path("init.png"), mask_image=Path("mask.png"), strength=0.6
    )
    extra = {
        "image": Image.new("RGB", (3000, 1500)),
        "mask_image": Image.new("L", (3000, 1500)),
    }
    t2i._fit_init(cond, extra, None)
    assert extra["image"].size == extra["mask_image"].size == (1024, 512)


def test_a_caller_chosen_frame_is_left_alone(t2i):
    """Seam erase and the tile sheets pass ``size`` and slice the result on their
    own grid; clamping there would hand back a picture of another size."""
    _frame, extra = _frame_after_fit(t2i, 2048, 2048, size=(2048, 2048))
    assert extra["image"].size == (2048, 2048)


def test_control_hint_matches_the_init_frame_when_img2img_and_controlnet_combine(
    tmp_path,
):
    """The Canny hint was letterboxed to a 1024 square while the frame was the
    reference's own (1024, 640): the bars and every edge sat off the picture."""
    wide = Image.new("RGB", (1000, 400), (0, 0, 0))
    for x in range(100, 900):
        for y in (50, 350):
            wide.putpixel((x, y), (255, 255, 255))
    frame = control.fit_init_frame(*wide.size)
    assert frame == (1000, 400)
    assert control.fit_init_frame(2000, 800) == (1024, 408)
    out = control.hint("canny", wide, size=frame)
    assert out.size == frame

    src = tmp_path / "ref.png"
    wide.save(src)
    dest = tmp_path / "control.png"
    meta = control.write_hint(src, dest, kind="canny", size=frame)
    with Image.open(dest) as im:
        assert im.size == frame
    assert meta["size"] == max(frame)


def test_q_generate_sizes_the_hint_to_the_init_frame(tmp_path, monkeypatch):
    """The wiring: ``_conditioning`` hands write_hint the frame, not the square."""
    import asyncio

    from realmspinner import _q_generate

    ref = tmp_path / "ref.png"
    Image.new("RGB", (2000, 800), (200, 200, 200)).save(ref)
    seen: dict = {}

    def fake_write_hint(src, dest, *, kind, size, **kw):
        seen["size"] = size
        return {"kind": kind, "size": 0, "edge_fraction": 0.0}

    monkeypatch.setattr(control, "write_hint", fake_write_hint)
    spec = models.BASE_MODELS["sdxl_cfg"]
    params = {"init_image": True, "control": "canny"}
    asyncio.run(
        _q_generate.GenerateOps._conditioning(
            types.SimpleNamespace(), tmp_path, params, spec
        )
    )
    assert seen["size"] == (1024, 408)


def test_a_raising_ip_adapter_load_still_runs_unload_ip_adapter_in_teardown(t2i, tmp_path):
    """diffusers registers the CLIP-ViT-H image encoder before the UNet weight
    load; when the load raises, teardown used to skip the unload and the encoder
    stayed attached for the life of the process."""
    adapter = models.IP_ADAPTERS["plus"]
    weights = tmp_path / adapter.dir_name / adapter.subfolder / adapter.weight_name
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8)).save(ref)

    t2i._pipe = MagicMock()
    t2i._pipe.load_ip_adapter.side_effect = OSError("unet weights unreadable")
    cond = Conditioning(ip_adapter="plus", ip_image=ref)
    with pytest.raises(OSError):
        t2i._conditioned(cond)
    t2i._pipe.unload_ip_adapter.assert_called_once()


def test_mask_with_controlnet_is_refused_before_the_controlnet_loads(t2i, tmp_path, monkeypatch):
    """The refusal depends on ``cond`` alone yet fired after the ~2.5 GB
    ControlNet load and ``from_pipe``; it now runs before ``self.load``."""
    loaded: list[str] = []
    monkeypatch.setattr(t2i, "load", lambda *_a, **_k: loaded.append("load"))
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            ControlNetModel=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: loaded.append("controlnet")
            )
        ),
    )
    cond = Conditioning(
        init_image=tmp_path / "i.png",
        mask_image=tmp_path / "m.png",
        control="canny",
        control_image=tmp_path / "h.png",
    )
    with pytest.raises(RuntimeError, match="a mask and a ControlNet cannot be combined"):
        t2i._generate("p", tmp_path / "out.png", conditioning=cond)
    assert loaded == []

    t2i._pipe = MagicMock()
    with pytest.raises(RuntimeError, match="a mask and a ControlNet cannot be combined"):
        t2i._conditioned(cond)
    assert loaded == []
