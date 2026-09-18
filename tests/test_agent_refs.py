"""``studio/modes/clay/agent/refs.py`` is the headless half of reference-image handling for an
MCP-driven Clay session: pure Pillow over bytes, no display anywhere near it.
Every test here builds its inputs with ``PIL.Image.new`` and a ``BytesIO``
rather than a fixture file, for the same reason the module itself takes bytes
in and bytes out -- nothing here should need a job directory, a running app,
or a GPU to prove itself.
"""

from __future__ import annotations

import ast
import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from warlock.service import files
from warlock.studio.modes.clay.agent import refs as agent_refs


def _png(size, mode="RGB", color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(size, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _noise_png(size):
    """A PNG that resists compression, so its encoded size actually reflects
    its pixel count -- a solid colour compresses to a few dozen bytes at any
    resolution, which would defeat ``test_bounded_png_halves_a_picture``'s
    monkeypatched ceiling no matter how large the image was."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buf, "PNG")
    return buf.getvalue()


def _giant_header_png() -> bytes:
    """A structurally valid PNG whose IHDR claims a huge width/height, built
    by patching the header of a genuine 1x1 PNG rather than allocating a real
    giant image. This is exactly the flat-20-MP-PNG shape ``files.to_png``
    guards against: the check reads ``im.width * im.height`` from the header
    before any pixel is decoded, so a fake header is enough to exercise the
    refusal without the cost the refusal exists to avoid.
    """
    data = bytearray(_png((1, 1)))
    assert data[12:16] == b"IHDR"
    width = height = 5000  # 25,000,000 px > files.MAX_IMAGE_PIXELS (16,000,000)
    struct.pack_into(">II", data, 16, width, height)
    crc = zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF
    struct.pack_into(">I", data, 29, crc)
    return bytes(data)


# --- normalise ----------------------------------------------------------


def test_normalise_shrinks_a_picture_past_the_stored_cap_and_reports_the_size_it_stored():
    raw = _png((2000, 1000))
    png, width, height = agent_refs.normalise(raw)
    assert max(width, height) == agent_refs.REFERENCE_MAX_SIDE
    assert (width, height) == (1024, 512)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (width, height)


def test_normalise_leaves_a_picture_already_under_the_cap_byte_for_byte_alone():
    raw = _png((100, 50))
    expected = files.to_png(raw)
    png, width, height = agent_refs.normalise(raw)
    assert png == expected
    assert (width, height) == (100, 50)


def test_normalise_keeps_alpha_only_when_the_source_had_it():
    jpeg = _jpeg((64, 64))
    png, _, _ = agent_refs.normalise(jpeg)
    with Image.open(io.BytesIO(png)) as im:
        assert im.mode == "RGB"

    rgba = _png((64, 64), mode="RGBA", color=(10, 20, 30, 128))
    png, _, _ = agent_refs.normalise(rgba)
    with Image.open(io.BytesIO(png)) as im:
        assert im.mode == "RGBA"


def test_normalise_refuses_a_picture_with_more_pixels_than_the_service_cap():
    with pytest.raises(files.ImageTooLarge):
        agent_refs.normalise(_giant_header_png())


# --- beside ---------------------------------------------------------------


def test_beside_puts_the_reference_left_and_the_render_right_at_one_height():
    size = 64
    red = _png((size, size), color=(255, 0, 0))
    blue = _png((size, size), color=(0, 0, 255))
    sheet = agent_refs.beside(red, blue, "reference", "render", size=size)
    with Image.open(io.BytesIO(sheet)) as im:
        assert im.size == (2 * size + agent_refs.GUTTER, agent_refs.LABEL_H + size)
        rgb = im.convert("RGB")
        left = rgb.getpixel((size // 2, agent_refs.LABEL_H + size // 2))
        right = rgb.getpixel((size + agent_refs.GUTTER + size // 2, agent_refs.LABEL_H + size // 2))
    assert left == (255, 0, 0)
    assert right == (0, 0, 255)


def test_beside_letterboxes_rather_than_stretching_an_oblong_reference():
    size = 100
    oblong = _png((200, 100), color=(255, 0, 0))  # 2:1, taller cell padding top/bottom
    filler = _png((size, size), color=(0, 255, 0))
    sheet = agent_refs.beside(oblong, filler, "reference", "render", size=size)
    with Image.open(io.BytesIO(sheet)) as im:
        rgb = im.convert("RGB")
        top_centre = rgb.getpixel((size // 2, agent_refs.LABEL_H + 2))
        middle = rgb.getpixel((size // 2, agent_refs.LABEL_H + size // 2))
    assert top_centre == (255, 255, 255)
    assert middle != (255, 255, 255)


# --- overlay ----------------------------------------------------------------


def test_overlay_at_alpha_zero_is_the_reference_and_at_one_is_the_render():
    size = 32
    red = _png((size, size), color=(255, 0, 0))
    blue = _png((size, size), color=(0, 0, 255))

    at_zero = agent_refs.overlay(red, blue, 0.0, size=size)
    with Image.open(io.BytesIO(at_zero)) as im:
        assert im.convert("RGB").getpixel((size // 2, size // 2)) == (255, 0, 0)

    at_one = agent_refs.overlay(red, blue, 1.0, size=size)
    with Image.open(io.BytesIO(at_one)) as im:
        assert im.convert("RGB").getpixel((size // 2, size // 2)) == (0, 0, 255)


def test_overlay_at_a_half_is_between_the_two():
    size = 32
    red = _png((size, size), color=(255, 0, 0))
    blue = _png((size, size), color=(0, 0, 255))
    mixed = agent_refs.overlay(red, blue, 0.5, size=size)
    with Image.open(io.BytesIO(mixed)) as im:
        r, g, b = im.convert("RGB").getpixel((size // 2, size // 2))
    assert 0 < r < 255
    assert 0 < b < 255
    assert g == 0


# --- bounded_png -------------------------------------------------------


def test_bounded_png_halves_a_picture_over_the_ceiling_exactly_once(monkeypatch):
    monkeypatch.setattr(agent_refs, "MAX_IMAGE_RESULT", 200)
    big = _noise_png((64, 48))
    assert len(big) > 200
    with Image.open(io.BytesIO(big)) as im:
        original_size = im.size
    shrunk = agent_refs.bounded_png(big)
    with Image.open(io.BytesIO(shrunk)) as im:
        assert im.size == (original_size[0] // 2, original_size[1] // 2)


def test_bounded_png_hands_a_picture_under_the_ceiling_back_unchanged():
    small = _png((16, 16))
    assert agent_refs.bounded_png(small) is small


# --- no display -------------------------------------------------------


def _outward(path: Path) -> set[str]:
    """Names this module reaches for outside itself, one hop deep. Modelled on
    ``tests/modes/sirens/test_sirens_imports.py``'s helper of the same name, which
    is the existing precedent in this codebase for pinning a headless
    package's outward imports by walking its own AST rather than trusting an
    import to fail loudly if it ever grew a GL dependency."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_nothing_here_needs_a_display():
    banned = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL"}
    path = Path(agent_refs.__file__)
    roots = {name.split(".")[0] for name in _outward(path)}
    assert not (roots & banned), f"studio/modes/clay/agent/refs.py imports {roots & banned}"
    assert not any(name.endswith("agent_clay") for name in _outward(path))
