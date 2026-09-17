"""Wall-clock budget for ABI 11's kernel: RotSprite.

Marked ``perf`` for ``test_perf_native_batch3.py``'s reason -- a wall-clock
reading taken while eight workers saturate the cores is a reading about the
scheduler, not the kernel. Run with ``uv run pytest -m perf -n 0``.

The batch 10 measurement (dev/measurements/2026-09-13-native-batch-10-candidates.md
S5) was 294 ms per move at 256^2 against a 16 ms gate; the floor here is a much
softer multiple of the numpy path, because the budget that actually matters
(under 16 ms) is a property of the shipped kernel doing its job at all, not a
ratio -- and a ratio floor survives a slower CI runner the way an absolute
millisecond count would not.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import transform as tf

pytestmark = pytest.mark.perf


def _timed(fn, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _without_native(fn):
    import os

    os.environ["WARLOCK_NATIVE"] = "0"
    native.reset()
    try:
        return fn()
    finally:
        os.environ.pop("WARLOCK_NATIVE", None)
        native.reset()


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_rotsprite_at_256_squared_is_at_least_four_times_the_numpy_path_rgba():
    rng = np.random.default_rng(0x907)
    plane = rng.integers(0, 255, size=(256, 256, 4), dtype=np.uint8)

    with_kernel = _timed(lambda: tf.rotsprite(plane, 30.0, expand=True))
    without = _without_native(lambda: _timed(lambda: tf.rotsprite(plane, 30.0, expand=True)))
    native.reset()

    assert without / with_kernel >= 4.0, (
        f"rotsprite rgba: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_rotsprite_at_256_squared_is_at_least_four_times_the_numpy_path_mask():
    rng = np.random.default_rng(0x908)
    plane = (rng.integers(0, 2, size=(256, 256)) * 255).astype(np.uint8)

    with_kernel = _timed(lambda: tf.rotsprite(plane, 30.0, expand=True))
    without = _without_native(lambda: _timed(lambda: tf.rotsprite(plane, 30.0, expand=True)))
    native.reset()

    assert without / with_kernel >= 4.0, (
        f"rotsprite mask: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


# --- flourish_smoke ------------------------------------------------------
#
# The batch 10 measurement (S4 of the same document) was 388 ms/frame for an
# 80-blob smoke layer against a 100 ms gate; the batch 11 prototype got that
# to 80.6 ms, a 4.6x speedup. The floor here is the same kind of softer
# multiple the rotsprite budget above states and for the same reason: what
# matters is the shipped kernel clearing the 100 ms gate at all, not the
# exact ratio, and 3x survives a slower CI runner a fixed millisecond count
# would not. Not read from dev/scripts/bench_native.py -- tests/ must never
# read dev/, since a public clone carries no dev/ at all -- so the fixture is
# rebuilt here, the same shape as that script's ``_case_smoke``.


def _smoke_layer(count: int):
    from warlock.studio.inker.flourish.recipe import Layer, Phase
    from warlock.studio.inker.flourish.render import FrameCtx

    ctx = FrameCtx(
        seed=2,
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


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_smoke_at_80_blobs_is_at_least_three_times_the_numpy_path():
    from warlock.studio.inker.flourish.prims import smoke

    layer, ctx = _smoke_layer(80)

    with_kernel = _timed(lambda: smoke.render(layer, ctx, None))
    without = _without_native(lambda: _timed(lambda: smoke.render(layer, ctx, None)))
    native.reset()

    assert without / with_kernel >= 3.0, (
        f"smoke: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )
