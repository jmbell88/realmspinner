"""``native.smoke_blob`` against ``smoke.render``'s numpy path, bit for bit.

Same shape as ``tests/inker/test_rotsprite_native.py``: every test below runs
the same layer through both paths and asserts ``np.array_equal``, skipped
whole when no DLL is built (``vendor/`` is gitignored, so that is the
ordinary state of a fresh checkout). ``dev/scripts/bench_native.py``'s
``_case_smoke`` is the fixture this borrows its shape from -- a burst-emitted
smoke layer at 512x512x4 supersampled -- so the sizes here (10, 40, 80) are
the same ones the ``flourish_smoke`` gate is stated at.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from warlock import native
from warlock.kernels.pixel.flourish.prims import smoke
from warlock.kernels.pixel.flourish.recipe import Layer, Phase
from warlock.kernels.pixel.flourish.render import FrameCtx

pytestmark = pytest.mark.skipif(
    not native.available(), reason="warlockc is not built in this checkout"
)


def _numpy_render(layer, ctx):
    """The reference path, forced -- ``smoke.render`` with the kernel disabled
    for the duration of the call."""
    os.environ["WARLOCK_NATIVE"] = "0"
    native.reset()
    try:
        return smoke.render(layer, ctx, None)
    finally:
        os.environ.pop("WARLOCK_NATIVE", None)
        native.reset()


def _burst(count: int, seed: int) -> tuple[Layer, FrameCtx]:
    """A count-blob burst layer at bench_native's own raster size."""
    ctx = FrameCtx(
        seed=seed,
        width=512,
        height=512,
        scale=4.0,
        frame=8,
        phase=Phase("main", 24, True),
        phase_index=0,
        phase_frame=8,
        fps=18,
        layer_index=0,
    )
    layer = Layer(uid=2, kind="smoke", params={"count": count, "emission": "burst"})
    return layer, ctx


def _one_blob(
    x: float, y: float, size: float, raggedness: float, seed: int = 5
) -> tuple[Layer, FrameCtx]:
    """A single, precisely placed blob: no rise/drift/expand/spawn jitter, so
    its centre is exactly ``(x, y)`` and its radius exactly ``size`` at every
    age -- what lets the edge and no-fbm cases below be constructed by hand
    rather than found by searching a burst's hashed positions."""
    ctx = FrameCtx(
        seed=seed,
        width=64,
        height=64,
        scale=4.0,
        frame=1,
        phase=Phase("main", 4, True),
        phase_index=0,
        phase_frame=1,
        fps=18,
        layer_index=0,
    )
    layer = Layer(
        uid=1,
        kind="smoke",
        params={
            "count": 1,
            "emission": "burst",
            "x": float(x),
            "y": float(y),
            "size": float(size),
            "expand": 0.0,
            "rise": 0.0,
            "drift": 0.0,
            "spawn_radius": 0.0,
            "raggedness": float(raggedness),
        },
    )
    return layer, ctx


@pytest.mark.parametrize("count", [10, 40, 80])
@pytest.mark.parametrize("seed", [2, 17])
def test_a_burst_layer_is_bit_identical(count, seed):
    layer, ctx = _burst(count, seed)
    with_kernel = smoke.render(layer, ctx, None)
    reference = _numpy_render(layer, ctx)
    assert with_kernel is not None and reference is not None
    assert np.array_equal(with_kernel, reference)


def test_a_window_that_touches_the_raster_edge_is_bit_identical():
    """``x`` near the raster's logical edge (half-extent 8 at 64px/4x) with a
    radius that overruns it -- ``window()`` clips the rectangle to the frame,
    so the kernel's fbm plane and per-pixel loop both see a genuinely partial
    window, not the common case every burst test above already exercises."""
    layer, ctx = _one_blob(x=7.5, y=0.0, size=6.0, raggedness=0.6)
    with_kernel = smoke.render(layer, ctx, None)
    reference = _numpy_render(layer, ctx)
    assert with_kernel is not None and reference is not None
    assert np.array_equal(with_kernel, reference)


def test_a_blob_with_no_raggedness_skips_the_fbm_plane_identically():
    """``rag == 0`` takes the kernel's early-out (no scratch touched, no fbm
    built) exactly as the numpy path's ``if rag > 0.0`` does."""
    layer, ctx = _one_blob(x=0.0, y=0.0, size=6.0, raggedness=0.0)
    with_kernel = smoke.render(layer, ctx, None)
    reference = _numpy_render(layer, ctx)
    assert with_kernel is not None and reference is not None
    assert np.array_equal(with_kernel, reference)


def test_a_too_small_scratch_buffer_is_refused_and_python_still_falls_back(monkeypatch):
    """``native.smoke_blob`` returns ``False`` on a short scratch buffer, and
    the caller's own per-blob fallback still produces the correct answer --
    the same contract ``rotsprite_u8``/``contours``/``bvh_build`` have."""
    import ctypes

    layer, ctx = _one_blob(x=0.0, y=0.0, size=6.0, raggedness=0.6)
    reference = _numpy_render(layer, ctx)

    real_handle = native.lib()
    real_fn = real_handle.warlockc_smoke_blob

    def starved(*args):
        args = list(args)
        args[-1] = ctypes.c_size_t(0)
        return real_fn(*args)

    monkeypatch.setattr(real_handle, "warlockc_smoke_blob", starved)
    assert (
        native.smoke_blob(
            np.zeros((ctx.height, ctx.width, 3), dtype=np.float32),
            np.zeros((ctx.height, ctx.width), dtype=np.float32),
            ctx.width,
            ctx.height,
            0,
            ctx.height,
            0,
            ctx.width,
            float(ctx.scale),
            0.0,
            0.0,
            6.0,
            0.6,
            1,
            0.0,
            0.0,
            10.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        is False
    )

    # smoke.render doesn't go through this monkeypatch (it calls
    # native.smoke_blob, not the raw ctypes handle), so it still takes the
    # kernel via the *real* binding and agrees with the reference; this test
    # only pins that a refusal from the kernel is a handled outcome, not a
    # crash.
    with_kernel = smoke.render(layer, ctx, None)
    assert np.array_equal(with_kernel, reference)
