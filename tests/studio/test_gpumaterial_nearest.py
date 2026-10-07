"""``GpuMaterial`` picks NEAREST sampling for a ``nearest`` material.

No GL context: textures are recorded by a fake ``ctx``. What is pinned is the
choice (filter, mipmaps, anisotropy) and that the shared texture cache cannot
hand a crisp texture to a smooth material or the reverse.
"""

from __future__ import annotations

import moderngl

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.viewer import scene as scenelib


class _Tex:
    def __init__(self) -> None:
        self.filter = None
        self.anisotropy = 1.0
        self.mipmaps = False
        self.released = False

    def build_mipmaps(self) -> None:
        self.mipmaps = True

    def release(self) -> None:
        self.released = True


class _Ctx:
    max_anisotropy = 16.0

    def __init__(self) -> None:
        self.made: list[_Tex] = []

    def texture(self, *_a, **_k) -> _Tex:
        self.made.append(_Tex())
        return self.made[-1]


def _pixels() -> tuple[int, int, bytes]:
    return (1, 1, b"\xff\x00\x00\xff")


def test_a_nearest_material_samples_nearest_with_no_mipmaps_or_anisotropy() -> None:
    ctx = _Ctx()

    mat = scenelib.GpuMaterial(ctx, gltf.Material(base_color=_pixels(), nearest=True), {})

    tex = mat.textures["base_color"]
    assert tex.filter == (moderngl.NEAREST, moderngl.NEAREST)
    assert tex.mipmaps is False
    assert tex.anisotropy == 1.0


def test_a_smooth_material_keeps_mipmapped_anisotropic_filtering() -> None:
    ctx = _Ctx()

    mat = scenelib.GpuMaterial(ctx, gltf.Material(base_color=_pixels()), {})

    tex = mat.textures["base_color"]
    assert tex.filter == (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    assert tex.mipmaps is True
    assert tex.anisotropy == 8.0


def test_two_materials_sharing_one_pixels_object_do_not_share_a_texture_across_nearest() -> None:
    ctx, cache, pixels = _Ctx(), {}, _pixels()

    crisp = scenelib.GpuMaterial(ctx, gltf.Material(base_color=pixels, nearest=True), cache)
    smooth = scenelib.GpuMaterial(ctx, gltf.Material(base_color=pixels), cache)
    crisp_again = scenelib.GpuMaterial(ctx, gltf.Material(base_color=pixels, nearest=True), cache)

    assert crisp.textures["base_color"] is not smooth.textures["base_color"]
    assert crisp.textures["base_color"].filter == (moderngl.NEAREST, moderngl.NEAREST)
    assert smooth.textures["base_color"].filter[0] == moderngl.LINEAR_MIPMAP_LINEAR
    # Same pixels, same choice: still one upload.
    assert crisp_again.textures["base_color"] is crisp.textures["base_color"]
    assert len(ctx.made) == 2


def test_nearest_applies_to_the_base_colour_slot_alone() -> None:
    ctx = _Ctx()
    material = gltf.Material(base_color=_pixels(), normal=(1, 1, b"\x80\x80\xff\xff"), nearest=True)

    mat = scenelib.GpuMaterial(ctx, material, {})

    assert mat.textures["normal"].filter == (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
