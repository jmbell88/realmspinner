"""``pipelines/imageout``: the web re-encodings a reference or a tile's own
picture can become.

``test_music_format.py``'s model, on the other pipeline: cheap, direct tests
of the conversion function itself rather than of the service layer that
derives it lazily (``tests/test_derive_2d.py`` covers that half -- staleness,
stage gating, the NotReady/Invalid split). None of these needs a ``svc``
fixture, a job directory, or a GPU; Pillow is already a core dependency, the
same way ``pipelines/audioout``'s tests need only ``soundfile``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from warlock.pipelines import imageout


def _save(tmp_path: Path, name: str, image: Image.Image) -> Path:
    path = tmp_path / name
    image.save(path, "PNG")
    return path


def test_convert_refuses_a_name_outside_its_format_table(tmp_path):
    # ``FORMATS`` is the allowlist ``audioout.convert`` follows for its own
    # names -- deriving a path from a caller-supplied name instead of looking
    # it up is exactly what an allowlist exists to prevent.
    source = _save(tmp_path, "input.png", Image.new("RGB", (8, 8), (10, 20, 30)))
    with pytest.raises(ValueError):
        imageout.convert(source, tmp_path / "out.bin", "input.bmp")


def test_a_webp_derived_from_an_rgba_picture_still_has_alpha(tmp_path):
    source = _save(
        tmp_path, "input.png", Image.new("RGBA", (16, 16), (10, 20, 30, 128))
    )
    out = tmp_path / "input.webp"
    imageout.convert(source, out, "input.webp")
    with Image.open(out) as im:
        assert im.mode == "RGBA"
        assert im.getpixel((0, 0)) == (10, 20, 30, 128)


def test_a_jpeg_asked_for_an_rgba_picture_is_refused_not_flattened(tmp_path):
    # JPEG cannot carry alpha. Flattening it onto black would be a silent lie
    # about the pixels the user asked for -- this refuses instead, and the
    # refusal names the control (field="format") the way ``NoSubject``'s
    # NotReady names the picture rather than saying "file not ready".
    source = _save(
        tmp_path, "input.png", Image.new("RGBA", (16, 16), (10, 20, 30, 128))
    )
    with pytest.raises(imageout.AlphaUnsupported) as caught:
        imageout.convert(source, tmp_path / "input.jpg", "input.jpg")
    # The address rides on the exception rather than on a ``service`` type,
    # because ``pipelines`` imports nothing from ``service``; the service layer
    # is what turns this into the ``Invalid`` the UI points at a control with,
    # and ``test_derive_2d`` is where that hop is pinned.
    assert caught.value.field == "format"
    assert not (tmp_path / "input.jpg").exists()


def test_a_palette_image_with_a_transparency_entry_is_also_refused_for_jpeg(tmp_path):
    # ``mode == "RGBA"`` alone misses this: a palette (P) image's alpha lives
    # in ``info["transparency"]`` rather than in a fourth channel, the same
    # case ``service.files.to_png`` already has to detect on the way in.
    im = Image.new("P", (16, 16))
    im.info["transparency"] = 0
    source = tmp_path / "input.png"
    im.save(source, "PNG")
    with pytest.raises(imageout.AlphaUnsupported) as caught:
        imageout.convert(source, tmp_path / "input.jpg", "input.jpg")
    assert caught.value.field == "format"


def test_a_jpeg_from_an_opaque_picture_is_written_as_rgb(tmp_path):
    source = _save(tmp_path, "input.png", Image.new("RGB", (16, 16), (10, 20, 30)))
    out = tmp_path / "input.jpg"
    imageout.convert(source, out, "input.jpg")
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"


def test_a_webp_from_an_opaque_picture_carries_no_alpha(tmp_path):
    # WEBP can encode alpha, but an opaque source should not grow one it never
    # had -- the same rule ``service.files.to_png`` keeps for the opposite
    # direction.
    source = _save(tmp_path, "input.png", Image.new("RGB", (16, 16), (10, 20, 30)))
    out = tmp_path / "input.webp"
    imageout.convert(source, out, "input.webp")
    with Image.open(out) as im:
        assert im.mode == "RGB"


def test_webp_is_lossless(tmp_path):
    # A checkerboard is the pattern lossy WebP compresses worst, so a mismatch
    # here would show up before it showed up on a real picture.
    im = Image.new("RGB", (8, 8))
    for y in range(8):
        for x in range(8):
            im.putpixel((x, y), (255, 0, 0) if (x + y) % 2 else (0, 255, 0))
    source = _save(tmp_path, "input.png", im)
    out = tmp_path / "input.webp"
    imageout.convert(source, out, "input.webp")
    with Image.open(out) as written:
        assert written.convert("RGB").tobytes() == im.tobytes()


def test_no_module_under_pipelines_imports_the_service_layer():
    """The rule `imageout` was briefly the only file in the package to break.

    ``pipelines/material.py`` and ``pipelines/pixel.py`` both state it in their
    own docstrings -- nothing here imports from ``service`` -- and until this
    test existed it was stated twice in prose and enforced nowhere. It matters
    for `imageout` specifically because the obvious way to refuse a JPEG with
    alpha is to raise ``service.errors.Invalid`` from inside the encoder, which
    is a layer inversion: `pipelines` also runs inside worker and Blender
    subprocesses, where importing the service layer drags a sqlite connection
    and the whole job store into a process that has no use for either.

    The shape that works instead is a ``ValueError`` defined in `pipelines`
    carrying its own ``field``, translated at the boundary by whichever service
    function called down -- ``imageout.AlphaUnsupported`` and the
    ``DERIVED_IMAGE`` arm of ``service/derive.get_file``.

    Scanned at module *and* function scope: a deferred import is still an
    import, and it is the one a reader is most likely to think does not count.
    """
    import ast

    root = Path(imageout.__file__).parent
    offenders: list[str] = []
    for module in sorted(root.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Relative: level 1 is a sibling in ``pipelines``, level 2 is
                # ``warlock.<x>`` -- ``service`` reached either way is the same
                # inversion.
                name = node.module or ""
                if name == "service" or name.startswith("service."):
                    offenders.append(f"{module.name}:{node.lineno} from ..{name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("warlock.service"):
                        offenders.append(f"{module.name}:{node.lineno} import {alias.name}")
    assert not offenders, f"pipelines reaches into service: {offenders}"
