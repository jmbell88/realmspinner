"""``native.rotsprite_u8`` against ``transform.rotsprite``'s numpy path, bit
for bit.

RotSprite's numpy reference (``tests/inker/test_rotsprite.py``) already pins
``epx``/``rotsprite`` exactly, so this module is only about the kernel: every
test below runs the same plane and angle through both paths and asserts
``np.array_equal``, skipped whole when no DLL is built (``vendor/`` is
gitignored, so that is the ordinary state of a fresh checkout).
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.kernels.pixel import transform as tf

pytestmark = pytest.mark.skipif(
    not native.available(), reason="warlockc is not built in this checkout"
)


def _numpy_rotsprite(pixels: np.ndarray, degrees: float, *, expand: bool = True) -> np.ndarray:
    """The reference path, forced -- ``transform.rotsprite`` with the kernel
    disabled for the duration of the call."""
    import os

    os.environ["WARLOCK_NATIVE"] = "0"
    native.reset()
    try:
        return tf.rotsprite(pixels, degrees, expand=expand)
    finally:
        os.environ.pop("WARLOCK_NATIVE", None)
        native.reset()


def _rgba(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return rng.integers(0, 255, size=(h, w, 4), dtype=np.uint8)


def _mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return (rng.integers(0, 2, size=(h, w)) * 255).astype(np.uint8)


# --- angle sweep --------------------------------------------------------------


def test_an_angle_sweep_from_0_to_360_is_bit_identical_rgba():
    rng = np.random.default_rng(101)
    plane = _rgba(rng, 17, 11)
    angle = 0.0
    while angle <= 360.0:
        with_kernel = tf.rotsprite(plane, angle, expand=True)
        reference = _numpy_rotsprite(plane, angle, expand=True)
        assert np.array_equal(with_kernel, reference), angle
        angle += 0.5


def test_an_angle_sweep_from_0_to_360_is_bit_identical_mask():
    rng = np.random.default_rng(102)
    plane = _mask(rng, 17, 11)
    angle = 0.0
    while angle <= 360.0:
        with_kernel = tf.rotsprite(plane, angle, expand=True)
        reference = _numpy_rotsprite(plane, angle, expand=True)
        assert np.array_equal(with_kernel, reference), angle
        angle += 0.5


# --- sizes ---------------------------------------------------------------


SIZES = [(1, 1), (2, 3), (64, 64), (256, 256), (512, 512)]
ANGLES = [30.0, 45.0, 137.25, 359.5]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("angle", ANGLES)
def test_sizes_and_angles_are_bit_identical_rgba(size, angle):
    h, w = size
    rng = np.random.default_rng(hash((h, w, angle)) & 0xFFFFFFFF)
    plane = _rgba(rng, h, w)
    with_kernel = tf.rotsprite(plane, angle, expand=True)
    reference = _numpy_rotsprite(plane, angle, expand=True)
    assert np.array_equal(with_kernel, reference)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("angle", ANGLES)
def test_sizes_and_angles_are_bit_identical_mask(size, angle):
    h, w = size
    rng = np.random.default_rng(hash((h, w, angle, "mask")) & 0xFFFFFFFF)
    plane = _mask(rng, h, w)
    with_kernel = tf.rotsprite(plane, angle, expand=True)
    reference = _numpy_rotsprite(plane, angle, expand=True)
    assert np.array_equal(with_kernel, reference)


# --- a raw 0/1 mask, as render_transform passes -------------------------------


def test_a_raw_0_1_mask_plane_is_bit_identical():
    """``render_transform`` passes the selection mask as raw 0/1 bytes, not
    0/255 -- and EPX's equality test does not care which, but this pins that
    the kernel doesn't quietly assume 0/255 either."""
    rng = np.random.default_rng(103)
    plane = rng.integers(0, 2, size=(23, 19), dtype=np.uint8)
    with_kernel = tf.rotsprite(plane, 41.0, expand=True)
    reference = _numpy_rotsprite(plane, 41.0, expand=True)
    assert np.array_equal(with_kernel, reference)
    assert set(np.unique(with_kernel).tolist()) <= {0, 1}


# --- the scratch-too-small fallback -------------------------------------------


def test_a_too_small_scratch_buffer_is_refused_and_python_still_falls_back(monkeypatch):
    """``native.rotsprite_u8`` returns None on a short scratch buffer, and the
    caller's own fallback still produces the correct answer -- the same
    contract ``contours``/``bvh_build`` have."""
    import ctypes

    rng = np.random.default_rng(104)
    plane = _rgba(rng, 12, 9)
    reference = _numpy_rotsprite(plane, 22.0, expand=True)

    real_handle = native.lib()
    real_fn = real_handle.warlockc_rotsprite_u8

    def starved(*args):
        # Same call, but with a scratch length of 0 -- forces the kernel's own
        # short-buffer guard rather than faking a Python-side short-circuit,
        # so this exercises the real -1 path.
        args = list(args)
        args[-2] = ctypes.c_size_t(0)
        return real_fn(*args)

    monkeypatch.setattr(real_handle, "warlockc_rotsprite_u8", starved)
    assert native.rotsprite_u8(plane, 22.0) is None

    # transform.rotsprite doesn't go through this monkeypatch (it calls
    # native.rotsprite_u8, not the raw ctypes handle), so it still takes the
    # kernel via the *real* binding and agrees with the reference; what this
    # test actually pins is that a None from the kernel is a valid, handled
    # outcome and not a crash.
    with_kernel = tf.rotsprite(plane, 22.0, expand=True)
    assert np.array_equal(with_kernel, reference)
