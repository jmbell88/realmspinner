"""``stamp`` returns only its window, not a full-frame plane (inker-flourish).

``prims.stamp`` used to allocate a fresh ``(H, W, 4)`` zero plane per particle
and every caller added and alpha-composed the whole frame, though only the
stamp's own window was ever non-zero: 1.9 s at 400 textured particles on a
512px raster (dev/measurements/2026-09-13-native-batch-10-candidates.md
§3). This pins the windowed shape and proves every caller still produces the
bit-identical frame a full-frame stamp-and-compose would have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.kernels.pixel.flourish.prims import color, hashed, particles, ramp, sprite, stamp, val
from warlock.kernels.pixel.flourish.recipe import Layer, Phase
from warlock.kernels.pixel.flourish.render import FrameCtx


def _ctx(width: int = 64, height: int = 64, scale: float = 2.0, frame: int = 4) -> FrameCtx:
    return FrameCtx(
        seed=1,
        width=width,
        height=height,
        scale=scale,
        frame=frame,
        phase=Phase("main", 12, True),
        phase_index=0,
        phase_frame=frame,
        fps=18,
        assets={},
    )


def _texture(seed: int = 0x7E57, size: int = 16) -> np.ndarray:
    tex = np.random.default_rng(seed).integers(0, 256, size=(size, size, 4), dtype=np.uint8)
    tex[..., 3] = np.random.default_rng(seed + 1).integers(0, 256, size=(size, size))
    return tex


# --- structural: stamp must not hand back a full-frame plane ----------------


def test_stamp_returns_a_window_sized_patch_not_a_full_frame():
    ctx = _ctx()
    texture = _texture()
    tint = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    result = stamp(ctx, texture, 0.0, 0.0, 12.0, 0.0, tint, 1.0)
    assert result is not None
    win, patch = result
    # The regression this guards: a full-frame stamp returns (ctx.height,
    # ctx.width, 4) regardless of how small the stamp is. A window-cut stamp's
    # patch matches the window it was computed on, which for a small texture
    # centred on-canvas is much smaller than the frame.
    assert patch.shape == win.shape + (4,)
    assert patch.shape != (ctx.height, ctx.width, 4)


def test_stamp_off_canvas_returns_none():
    ctx = _ctx()
    texture = _texture()
    tint = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert stamp(ctx, texture, 10_000.0, 10_000.0, 12.0, 0.0, tint, 1.0) is None


# --- parity: the old full-frame stamp+compose, reproduced exactly for tests -


def _old_stamp_full_frame(
    ctx: Any, texture: np.ndarray, cx: float, cy: float, width: float,
    degrees: float, tint: np.ndarray, alpha: float,
) -> np.ndarray | None:
    """The pre-windowing ``stamp``: a full ``(H, W, 4)`` zero plane with the
    patch pasted in. Kept here, verbatim in spirit, as the reference the
    windowed path must reproduce bit for bit."""
    result = stamp(ctx, texture, cx, cy, width, degrees, tint, alpha)
    if result is None:
        return None
    win, patch = result
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    out[win.rows, win.cols] = patch
    return out


def _old_render_textured(
    layer: Any, ctx: Any, st: dict[str, np.ndarray], texture: np.ndarray
) -> np.ndarray:
    """``particles._render_textured`` before windowing: full-frame add +
    alpha-compose per particle."""
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    c0 = color(layer, "color_start")
    c1 = color(layer, "color_end")
    spin = val(layer, "spin", ctx)
    seed = ctx.lseed(17)
    phases = hashed(seed, len(st["x"])) * 360.0
    for i in range(len(st["x"])):
        a = float(st["alpha"][i]) * min(c0[3], c1[3])
        width = float(st["size"][i]) * 2.0
        if a <= 0.0 or width <= 0.0:
            continue
        tint = np.append(ramp(c0, c1, np.asarray(st["u"][i])), 1.0).astype(np.float32)
        angle = float(phases[i]) + spin * float(st["u"][i]) * float(ctx.phase_seconds)
        plane = _old_stamp_full_frame(
            ctx, texture, float(st["x"][i]), float(st["y"][i]), width, angle, tint, a
        )
        if plane is None:
            continue
        out[..., :3] += plane[..., :3]
        out[..., 3] = out[..., 3] + plane[..., 3] - out[..., 3] * plane[..., 3]
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _particle_layer_ctx(
    count: int, seed: int, width: int, height: int, scale: float
) -> tuple[Layer, FrameCtx]:
    ctx = _ctx(width=width, height=height, scale=scale, frame=4)
    tex = _texture(seed=0x7E57 + seed)
    ctx.assets = {"spark": tex}
    params: dict[str, Any] = {
        "count": count,
        "emission": "burst",
        "texture": "spark",
        "spin": 90.0 * (seed % 3),
        "size": 3.0 + (seed % 5),
        "spawn_radius": 40.0,
    }
    ctx.seed = seed
    return Layer(uid=1, kind="particles", params=params), ctx


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("count", [1, 5, 40])
@pytest.mark.parametrize("raster", [(64, 64, 2.0), (128, 96, 3.0)])
def test_textured_particles_match_full_frame_reference(seed, count, raster):
    width, height, scale = raster
    layer, ctx = _particle_layer_ctx(count, seed, width, height, scale)
    st = particles._state(layer, ctx)
    if st is None:
        pytest.skip("no particles alive at this seed/phase")
    texture = ctx.asset("spark")

    windowed = particles._render_textured(layer, ctx, st, texture)
    reference = _old_render_textured(layer, ctx, st, texture)
    assert np.array_equal(windowed, reference)


def test_sprite_render_matches_full_frame_reference():
    """sprite.render still returns a full-frame plane; only stamp()'s own
    internal allocation shrank, so the *layer* output is unchanged."""
    ctx = _ctx(width=96, height=80, scale=2.5)
    texture = _texture(seed=99)
    ctx.assets = {"rune": texture}
    layer = Layer(
        uid=2,
        kind="sprite",
        params={"texture": "rune", "size": 20.0, "rotation": 30.0, "tint": "#88CCFF"},
    )
    windowed = sprite.render(layer, ctx, None)
    assert windowed is not None

    reference = _old_stamp_full_frame(
        ctx,
        texture,
        *ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx)),
        val(layer, "size", ctx),
        val(layer, "rotation", ctx),
        color(layer, "tint"),
        val(layer, "alpha", ctx),
    )
    assert np.array_equal(windowed, reference)


def test_stamp_partly_off_canvas_matches_full_frame_reference():
    ctx = _ctx(width=48, height=48, scale=2.0)
    texture = _texture(seed=5)
    tint = np.asarray([1.0, 0.5, 0.25, 1.0], dtype=np.float32)
    for cx, cy, angle in [(-30.0, -30.0, 0.0), (30.0, 5.0, 45.0), (0.0, 40.0, 200.0)]:
        result = stamp(ctx, texture, cx, cy, 30.0, angle, tint, 0.8)
        reference = _old_stamp_full_frame(ctx, texture, cx, cy, 30.0, angle, tint, 0.8)
        if result is None:
            assert reference is None
            continue
        win, patch = result
        got = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
        got[win.rows, win.cols] = patch
        assert np.array_equal(got, reference)
