"""Regression tests closing the 2026-10-14 audit's Packwright findings handled by F38
(packwright-01, -02, -08, -09, -10, -12, -14, -15, -18).

Each test's name is the claim.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.engine import rpack
from realmspinner.studio.modes.packwright.engine.document import PackDoc
from realmspinner.studio.modes.packwright.engine.sources import (
    Sprite,
    dedup_tiles,
)

from .test_packwright_mode import FakeCtx, _pack, _tab
from .test_packwright_mode import _sprite as _mode_sprite
from .test_rpack import _doc, _rewrite, _rewrite_member

_ROOT = Path(__file__).resolve().parents[3]
_ENGINE = _ROOT / "src" / "realmspinner" / "studio" / "modes" / "packwright" / "engine"


def _sprite(key: str, name: str = "name") -> Sprite:
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (10, 20, 30, 255)
    return Sprite(key=key, name=name, pixels=pixels)


def _zip_of(members: dict[str, bytes], *, stored: bool = False) -> bytes:
    out = io.BytesIO()
    kind = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(out, "w", kind) as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out.getvalue()


def _members(doc: PackDoc) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(rpack.rpack_bytes(doc))) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


# --- packwright-18 ------------------------------------------------------------


def test_packwright_engine_docstrings_name_modules_that_exist():
    """The stale spellings the audit found are gone from the engine's prose, and
    every ``studio.modes...``/``kernels...`` module a docstring names resolves."""
    stale = [
        "studio/inker/",
        "studio/clay/",
        "studio/plotter/",
        "studio.pixelguard",
        "clay/serialize.py",
        "~..grid2d.tileset",
    ]
    for path in (
        _ENGINE / "__init__.py",
        _ENGINE / "rpack.py",
        _ENGINE / "tsxout.py",
    ):
        text = path.read_text(encoding="utf-8")
        for word in stale:
            assert word not in text, f"{path.name} still names {word!r}"

    # Every dotted ``realmspinner.x.y`` module a docstring cites must exist.
    for path in (_ENGINE / "__init__.py", _ENGINE / "rpack.py", _ENGINE / "tsxout.py"):
        text = path.read_text(encoding="utf-8")
        for dotted in re.findall(r"realmspinner(?:\.[a-z_0-9]+)+", text):
            parts = dotted.split(".")[1:]
            base = _ROOT / "src" / "realmspinner"
            probe = base.joinpath(*parts)
            # A trailing attribute (a constant, say) is allowed: walk up until
            # a module or package is found.
            while parts and not (
                base.joinpath(*parts).with_suffix(".py").exists()
                or base.joinpath(*parts).is_dir()
            ):
                parts = parts[:-1]
            assert parts, f"{path.name} names {dotted}, which resolves to nothing ({probe})"


# --- packwright-09 ------------------------------------------------------------


def test_add_source_refuses_a_display_name_read_rpack_would_refuse():
    doc = PackDoc()
    with pytest.raises(ValueError, match="control character"):
        doc.add_source(_sprite("a", name="bad\x01name"))
    with pytest.raises(ValueError, match="255"):
        doc.add_source(_sprite("b", name="x" * 256))
    assert doc.sources == []
    # The slash the reader deliberately keeps is still welcome.
    doc.add_source(_sprite("c", name="layer/one"))
    good = doc.add_source(_sprite("d", name="x" * 255))
    with pytest.raises(ValueError, match="control character"):
        doc.replace_source(good.uid, _sprite("d", name="tab\tname"))
    # And what the door lets in the reader reopens.
    assert [s.key for s in rpack.read_rpack(rpack.rpack_bytes(doc)).sources] == ["c", "d"]


# --- packwright-01 ------------------------------------------------------------


def test_read_rpack_refuses_a_crc_damaged_member_and_a_deeply_nested_manifest_as_value_error():
    members = _members(_doc(2))
    # A stored member whose bytes changed after its CRC was written.
    data = bytearray(_zip_of(members, stored=True))
    blob = members["sources/0.png"]
    at = bytes(data).index(blob)
    data[at + len(blob) // 2] ^= 0xFF
    with pytest.raises(ValueError, match="sources/0.png|not an image|atlas"):
        rpack.read_rpack(bytes(data))

    # A manifest nested past the interpreter's recursion limit.
    deep = b"[" * 200_000 + b"]" * 200_000
    hostile = dict(members)
    hostile[rpack.MANIFEST] = deep
    with pytest.raises(ValueError, match="atlas document"):
        rpack.read_rpack(_zip_of(hostile))


# --- packwright-08 ------------------------------------------------------------


def _png_then_zeros(zeros: int) -> bytes:
    from realmspinner.studio.modes.plotter.engine.pngio import png_bytes

    return png_bytes(np.full((2, 2, 4), 255, dtype=np.uint8)) + b"\0" * zeros


def test_read_rpack_refuses_two_entries_that_name_one_image_member():
    doc = _doc(3)
    data = _rewrite(
        doc,
        lambda m: [e.update(image="sources/0.png") for e in m["sources"]],
    )
    with pytest.raises(ValueError, match="more than one source|sources/0.png"):
        rpack.read_rpack(data)


def test_read_rpack_refuses_a_source_member_larger_than_any_legal_image_could_be():
    members = _members(_doc(2))
    members["sources/0.png"] = _png_then_zeros(rpack.MAX_SOURCE_PIXELS * 4 + (32 << 20))
    with pytest.raises(ValueError, match="sources/0.png"):
        rpack.read_rpack(_zip_of(members))


# --- packwright-10 ------------------------------------------------------------


def test_read_rpack_refuses_an_oversized_manifest_before_parsing_it():
    members = _members(_doc(2))
    # Valid JSON, padded with whitespace past any manifest a document can write.
    members[rpack.MANIFEST] = members[rpack.MANIFEST] + b" " * (rpack.MAX_MANIFEST_BYTES + 1)
    with pytest.raises(ValueError, match="manifest"):
        rpack.read_rpack(_zip_of(members))


def test_read_rpack_refuses_more_slices_than_any_document_can_hold():
    count = rpack.MAX_SLICES_PER_DOCUMENT + 1

    def many(manifest):
        manifest["sources"][0]["slices"] = [
            {"name": "", "bounds": {"x": 0, "y": 0, "w": 1, "h": 1}}
        ] * count

    members = _members(_doc(2))
    manifest = json.loads(members[rpack.MANIFEST])
    many(manifest)
    members[rpack.MANIFEST] = json.dumps(manifest, separators=(",", ":")).encode()
    assert len(members[rpack.MANIFEST]) < rpack.MAX_MANIFEST_BYTES
    with pytest.raises(ValueError, match="slices"):
        rpack.read_rpack(_zip_of(members))


# --- packwright-15 ------------------------------------------------------------


def test_read_rpack_refuses_a_member_that_is_not_a_png():
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(out, format="BMP")
    data = _rewrite_member(_doc(2), "sources/0.png", out.getvalue())
    with pytest.raises(ValueError, match="not an image this build can read"):
        rpack.read_rpack(data)


def test_pixelguard_opened_never_hands_back_an_eps_image(tmp_path):
    """EPS is the one Pillow format whose ``load`` starts Ghostscript."""
    from realmspinner.core.safeio import pixelguard

    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 4 4\n%%EndComments\nshowpage\n"
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError), pixelguard.opened(
        io.BytesIO(eps), "a test image"
    ):
        pass


def test_pixelguard_opened_can_be_told_to_accept_png_only():
    from PIL import Image

    from realmspinner.core.safeio import pixelguard

    out = io.BytesIO()
    Image.new("RGBA", (2, 2)).save(out, format="BMP")
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError), pixelguard.opened(
        io.BytesIO(out.getvalue()), "a layer", formats=["PNG"]
    ):
        pass
    # Without the restriction the same BMP is an ordinary image.
    with pixelguard.opened(io.BytesIO(out.getvalue()), "a layer") as image:
        assert image.format == "BMP"


# --- packwright-14 ------------------------------------------------------------


def test_dedup_tiles_never_merges_two_tiles_that_are_not_dihedral_images_of_each_other():
    # a (2 rows x 3 cols) tile and a (3 rows x 2 cols) tile whose raw bytes are
    # the same flat sequence: not images of one another under any flip or turn.
    flat = np.arange(24, dtype=np.uint8).reshape(6, 4)
    wide = np.ascontiguousarray(flat.reshape(2, 3, 4))
    tall = np.ascontiguousarray(flat.reshape(3, 2, 4))
    sprites = [
        Sprite(key="wide", name="wide", pixels=wide),
        Sprite(key="tall", name="tall", pixels=tall),
    ]
    for orientations in (True, False):
        kept, dropped = dedup_tiles(sprites, orientations=orientations)
        assert dropped == 0 and [s.key for s in kept] == ["wide", "tall"]
    # The thing the key exists for still works: a turned copy is a duplicate.
    turned = Sprite(key="turned", name="turned", pixels=np.ascontiguousarray(wide[::-1, ::-1]))
    kept, dropped = dedup_tiles([sprites[0], turned], orientations=True)
    assert dropped == 1


# --- packwright-02 ------------------------------------------------------------


def test_save_as_refuses_a_path_another_open_atlas_tab_holds(tmp_path, monkeypatch):
    from realmspinner.service.errors import ServiceError
    from realmspinner.studio import dialogs

    ctx = FakeCtx()
    first = _tab(ctx)
    held = tmp_path / "atlas.rpack"
    held.write_bytes(b"the first tab's file")
    first.path = held
    second = packwright_mode.new_document(ctx)
    second.doc.add_source(_mode_sprite("s9"))
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: held)

    with pytest.raises(ServiceError, match="already open"):
        packwright_mode.save_as(ctx, second)

    assert held.read_bytes() == b"the first tab's file"
    assert second.path != held

    # A tab saving over its own path is not a conflict.
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: held)
    packwright_mode.save_as(ctx, first)
    assert zipfile.is_zipfile(held)


# --- packwright-12 ------------------------------------------------------------


class _OnceCtx(FakeCtx):
    """A ctx with the real shell's ``toast_once`` shape: coalesced by text."""

    def __init__(self) -> None:
        super().__init__()
        self.once: list[str] = []

    def toast_once(self, text, kind="info", action=None, action_arg=None) -> None:
        if text not in self.once:
            self.once.append(text)


def test_repeated_export_refusals_coalesce_into_one_toast():
    ctx = _OnceCtx()
    tab = _tab(ctx)
    # Nothing packed yet, then a pack that is behind the document.
    for _ in range(3):
        packwright_mode.export_files(ctx, tab)
        packwright_mode.export_library(ctx, tab)
    _pack(ctx, tab)
    tab.pack_dirty = True
    for _ in range(3):
        packwright_mode.export_files(ctx, tab)
        packwright_mode.export_library(ctx, tab)
    tab.pack_dirty = False
    tab.pack_error = "The pack is out of date."
    packwright_mode.export_files(ctx, tab)
    packwright_mode.export_library(ctx, tab)

    assert [t for t in ctx.toasts if t[1] == "error"] == []
    assert "Nothing packed yet." in ctx.once
    assert "Still packing your latest edits -- try again in a moment." in ctx.once
    assert tab.pack_stale_why in ctx.once


def test_a_zlib_corrupt_deflate_member_is_a_named_refusal():
    """The deflate half of packwright-01: a *compressed* member whose stream is
    damaged raises ``zlib.error``, which is no more a ``ValueError``."""
    members = _members(_doc(2))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(rpack.MANIFEST, members[rpack.MANIFEST] * 20)
        zf.writestr("sources/0.png", members["sources/0.png"])
    data = bytearray(out.getvalue())
    header = bytes(data).index(b"pack.json") + len(b"pack.json")
    for i in range(header + 8, header + 12):
        data[i] ^= 0xFF
    with pytest.raises(ValueError):
        rpack.read_rpack(bytes(data))
