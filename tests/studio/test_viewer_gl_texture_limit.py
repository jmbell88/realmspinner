"""F20-12: ``gltf.MAX_TEXTURE_SIDE`` is a fixed 16,384, but a card's own
``GL_MAX_TEXTURE_SIZE`` can be smaller; the upload asks the context and skips the
texture (the loader's cosmetic-loss policy) rather than failing in ``ctx.texture``
on the frame thread and taking the whole model (create-38)."""

from __future__ import annotations

from types import SimpleNamespace

import moderngl

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.viewer import scene as scenelib


class _Texture:
    filter = None
    anisotropy = 1.0

    def build_mipmaps(self) -> None:
        pass

    def release(self) -> None:
        pass


class _Ctx:
    max_anisotropy = 1.0

    def __init__(self, limit: int) -> None:
        self.info = {"GL_MAX_TEXTURE_SIZE": limit}
        self.sizes: list[tuple[int, int]] = []

    def texture(self, size, components, data):
        self.sizes.append(size)
        if max(size) > self.info["GL_MAX_TEXTURE_SIZE"]:
            raise moderngl.Error("texture too large")
        return _Texture()


def _material(width: int, height: int) -> gltf.Material:
    return gltf.Material(base_color=(width, height, b"\0" * 4))


def test_a_texture_past_this_cards_gl_limit_is_skipped_not_uploaded():
    ctx = _Ctx(limit=4096)

    mat = scenelib.GpuMaterial(ctx, _material(8192, 16), {})

    assert ctx.sizes == [], "the oversize map must never reach ctx.texture"
    assert "base_color" not in mat.textures and mat.defines == []


def test_a_texture_within_this_cards_gl_limit_still_uploads():
    ctx = _Ctx(limit=4096)

    mat = scenelib.GpuMaterial(ctx, _material(4096, 16), {})

    assert ctx.sizes == [(4096, 16)]
    assert "base_color" in mat.textures


def test_a_context_that_reports_no_limit_uploads_as_before():
    ctx = SimpleNamespace(max_anisotropy=1.0, sizes=[])
    ctx.texture = lambda size, components, data: (ctx.sizes.append(size), _Texture())[1]

    scenelib.GpuMaterial(ctx, _material(32, 32), {})

    assert ctx.sizes == [(32, 32)]
