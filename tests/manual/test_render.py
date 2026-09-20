"""``manual/render.py``'s image block, and the cache it draws through.

The 2026-09-18 audit, finding shell-04: ``ThumbnailCache.get`` decoded every
PNG on the frame thread, and its own docstring called that "one small PNG" --
true for a 256 px library thumbnail but not for the manual's 1600 px
screenshots, measured at 12-21ms per decode (most of a 60fps frame budget).
"""

from __future__ import annotations

import time

from PIL import Image

from realmspinner.studio.textures import ThumbnailCache


class _FakeTexture:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.released = False
        self.glo = id(self)

    def release(self):
        self.released = True


class _FakeGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size, components, data):
        return _FakeTexture(size)


def _png(path, size=(64, 64), color=(10, 20, 30, 255)):
    Image.new("RGBA", size, color).save(path)


def test_thumbnail_cache_manual_screenshot_decode_does_not_block_frame_thread(tmp_path):
    """``background=True`` returns None immediately rather than decoding
    inline -- the whole point being that the caller (``_draw_image``) gets its
    answer without waiting on PIL. Without the fix this call decodes
    synchronously and hands back a texture on the very first call.
    """
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "screenshot.png"
    _png(path, size=(1600, 950))

    started = time.perf_counter()
    texture = cache.get("manual:ch.png", path, max_side=1600, background=True)
    elapsed = time.perf_counter() - started

    assert texture is None
    # A generous bound: an inline decode of a real 1600px PNG measured
    # 12-21ms on the audit's machine, so even a slow disk should not make a
    # *submit-and-return* call take anywhere near that.
    assert elapsed < 0.005


def test_a_background_decode_lands_on_a_later_frame(tmp_path):
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "screenshot.png"
    _png(path, size=(200, 100))

    first = cache.get("manual:ch.png", path, max_side=1600, background=True)
    assert first is None

    # The decode pool runs on real threads; give it a moment to land, the way
    # the frame loop's next ``begin_frame`` naturally would a frame later.
    deadline = time.monotonic() + 2.0
    texture = None
    while time.monotonic() < deadline and texture is None:
        cache.begin_frame()
        texture = cache.get("manual:ch.png", path, max_side=1600, background=True)
        if texture is None:
            time.sleep(0.01)

    assert texture is not None
    assert texture.size == (200, 100)


def test_a_missing_path_degrades_to_alt_text_without_ever_going_background(tmp_path):
    """A path that does not exist fails the ``stat()`` before ``background``
    is even consulted, the same as the synchronous path -- there is nothing
    to decode, so nothing to hand to the pool."""
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "does-not-exist.png"

    texture = cache.get("manual:missing.png", path, max_side=1600, background=True)

    assert texture is None
    assert not cache._inflight
