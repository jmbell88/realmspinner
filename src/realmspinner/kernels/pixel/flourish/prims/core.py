"""The core: a radial gradient with a pulse and a noisy edge.

White-hot centre through the inner colour to the outer colour to nothing. The
edge is eroded by fbm so it reads as a ball of fire or energy rather than a
lens flare, and the radius breathes with ``pulse``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, color, fbm_plane, premultiply, ramp, val, window

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "radius": Param("curve", 18.0, 0.0, 512.0, label="px"),
    "color_inner": Param("color", "#FFF1B0"),
    "color_outer": Param("color", "#FF6A1A"),
    "softness": Param("float", 0.6, 0.0, 1.0, label="edge fade"),
    "intensity": Param("curve", 1.0, 0.0, 2.0),
    "pulse": Param("float", 0.12, 0.0, 1.0, label="radius swing"),
    "pulse_hz": Param("float", 6.0, 0.0, 30.0),
    "noise": Param("float", 0.35, 0.0, 1.0, label="edge erosion"),
    "noise_scale": Param("float", 8.0, 1.0, 128.0, label="px per cell"),
    "noise_speed": Param("float", 2.0, 0.0, 20.0),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    radius = val(layer, "radius", ctx)
    pulse = val(layer, "pulse", ctx)
    if pulse > 0.0:
        radius *= 1.0 + pulse * float(np.sin(2.0 * np.pi * val(layer, "pulse_hz", ctx) * ctx.time))
    if radius <= 0.0:
        return None
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    amount = val(layer, "noise", ctx)
    # The 2026-09-26 audit, finding inker-flourish-04: the erosion below scales
    # ``d`` (distance/radius) by ``1 + (n-0.5)*2*amount``, so the *smallest*
    # factor it can apply is ``1 - amount`` -- and a visible pixel is wherever
    # the eroded ``d`` still reaches 1, i.e. a raw distance of up to
    # ``radius / (1 - amount)``, not ``radius * (1 + amount)`` as this window
    # sized it. The two agree to first order for a small ``amount`` and
    # diverge fast approaching 1 (2x vs 10x at ``amount=0.9``), clipping the
    # eroded edge to a square at high noise. Floored so ``amount`` at its own
    # ceiling of 1.0 cannot divide by zero.
    win = window(ctx, cx, cy, radius / max(1.0 - amount, 0.01))
    if win is None:
        return None
    d = np.sqrt((win.x - cx) ** 2 + (win.y - cy) ** 2) / np.float32(radius)
    if amount > 0.0:
        scale = val(layer, "noise_scale", ctx)
        drift = val(layer, "noise_speed", ctx) * ctx.time
        n = fbm_plane(ctx, ctx.lseed(), scale=scale, dy=-drift, octaves=3, win=win)
        d = d * (1.0 + (n - 0.5) * 2.0 * amount)

    softness = max(val(layer, "softness", ctx), 0.02)
    alpha = np.clip((1.0 - d) / softness, 0.0, 1.0).astype(np.float32)
    alpha *= alpha  # a rounder falloff than linear
    inner = color(layer, "color_inner")
    outer = color(layer, "color_outer")
    rgb = ramp(inner, outer, d)
    white = np.clip(1.0 - d * 2.5, 0.0, 1.0)[..., None]
    rgb = rgb + (1.0 - rgb) * white * 0.8
    intensity = val(layer, "intensity", ctx)
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    out[win.rows, win.cols] = premultiply(
        np.clip(rgb * intensity, 0.0, 1.0), alpha * min(inner[3], outer[3])
    )
    return out
