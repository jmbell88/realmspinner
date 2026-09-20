"""Sprite metadata through Packwright: pivots and named rectangles.

The whole of S5 is plumbing -- the packer decides nothing from any of it -- so
what these check is that it *arrives* and that it costs nothing when it is
absent. Three seams are worth naming.

The **duck-typed read**: Packwright takes metadata off a document through one
method it never introspects, so it can go on knowing nothing about frames, keys
or slices and its import pin stays honest.

The **rename rebuild**: ``PackDoc.sprites`` reconstructs a renamed sprite field
by field, which is the one place a carried-through field can silently stop being
carried through.

**Additive means byte-identical**: an atlas of sprites with no metadata, and a
``.rpack`` of the same, are exactly what they were.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from realmspinner.studio.modes.packwright.engine import rpack, texturepacker
from realmspinner.studio.modes.packwright.engine.document import PackDoc
from realmspinner.studio.modes.packwright.engine.layout import PackSettings, layout
from realmspinner.studio.modes.packwright.engine.sources import (
    EMPTY_META,
    SliceSpec,
    Sprite,
    SpriteMeta,
    sprite_meta,
    sprites_from_document,
)

PANEL = SliceSpec(name="panel", x=2, y=3, w=8, h=6, pivot=(6.0, 9.0), center=(4, 5, 4, 2))


def _pixels(w: int = 16, h: int = 12, mark=(2, 2, 10, 8)) -> np.ndarray:
    plane = np.zeros((h, w, 4), dtype=np.uint8)
    plane[mark[1] : mark[3], mark[0] : mark[2]] = (255, 0, 0, 255)
    return plane


def _sprite(key: str = "a", meta: SpriteMeta = EMPTY_META) -> Sprite:
    return Sprite(key=key, name=key, pixels=_pixels(), meta=meta)


# --- the frozen types ---------------------------------------------------------


def test_a_sprite_with_no_metadata_is_the_sprite_this_packer_always_had():
    sprite = Sprite(key="a", name="a", pixels=_pixels())
    assert sprite.meta is EMPTY_META
    assert not sprite.meta


def test_plain_data_is_coerced_into_the_frozen_types():
    meta = sprite_meta(
        {
            "pivot": (6.0, 9.0),
            "slices": [
                {"name": "panel", "x": 2, "y": 3, "w": 8, "h": 6,
                 "pivot": (6.0, 9.0), "center": (4, 5, 4, 2)}
            ],
        }
    )
    assert meta == SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,))


def test_a_slice_that_will_not_parse_costs_the_slice_and_not_the_sprite():
    """This is a read of somebody else's data, so it is tolerant -- ``.rpack``'s
    reader is the one that refuses, because there a malformed field means a file
    that is wrong about itself."""
    meta = sprite_meta({"slices": [{"name": "bad"}, {"name": "ok", "x": 0, "y": 0,
                                                     "w": 4, "h": 4}]})
    assert [one.name for one in meta.slices] == ["ok"]


def test_anything_that_is_not_a_mapping_is_nothing_at_all():
    for raw in (None, "meta", 7, []):
        assert sprite_meta(raw) is EMPTY_META or sprite_meta(raw) == EMPTY_META


# --- the duck-typed read ------------------------------------------------------


class _Layer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.pixels = _pixels()


class _Doc:
    """Enough of a document to enumerate, plus the one method Packwright asks
    for -- which is the whole of the interface."""

    def __init__(self, per_frame: dict | None = None) -> None:
        self.anim = None
        self.stack = [_Layer("bg"), _Layer("ink")]
        self.asked: list = []
        self._per_frame = per_frame or {}

    def sprite_meta_for_frame(self, frame_uid):
        self.asked.append(frame_uid)
        return self._per_frame.get(frame_uid) or {
            "pivot": (6.0, 9.0),
            "slices": [{"name": "panel", "x": 2, "y": 3, "w": 8, "h": 6}],
        }


def test_a_still_documents_layers_share_the_documents_metadata():
    """A slice is a rectangle on the canvas, not on a layer."""
    doc = _Doc()
    sprites = sprites_from_document(doc, prefix="d")
    # Read once, for the document, not once per layer.
    assert doc.asked == [None]
    assert len(sprites) == 2
    assert all(one.meta.pivot == (6.0, 9.0) for one in sprites)


def test_a_document_with_nothing_to_say_is_not_an_error():
    class _Bare:
        anim = None
        stack = (_Layer("bg"), _Layer("ink"))

    doc = _Bare()
    assert not hasattr(doc, "sprite_meta_for_frame")
    assert all(one.meta is EMPTY_META for one in sprites_from_document(doc, prefix="d"))


def test_an_inker_document_hands_its_slices_over_per_frame():
    """The real seam, end to end: the resolving happens on the editor's side and
    what crosses is plain data."""
    from realmspinner.kernels.pixel.document import Document
    from realmspinner.kernels.pixel.slices import SliceKey

    doc = Document.blank(16, 12)
    doc.add_frame()
    entry = doc.add_slice((2, 3, 10, 9), name="panel", pivot=(4.0, 6.0))
    doc.set_slice_key(
        entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(0, 0, 4, 4))
    )
    sprites = sprites_from_document(doc, prefix="d")
    assert [one.meta.slices[0].x for one in sprites] == [2, 0]
    assert sprites[0].meta.pivot == (6.0, 9.0)
    assert sprites[1].meta.pivot is None


# --- through the layout -------------------------------------------------------


def test_the_layout_carries_the_metadata_onto_its_frames():
    sprites = [_sprite("a", SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,)))]
    result = layout(sprites, PackSettings(power_of_two=False))
    assert result.frames[0].pivot == (6.0, 9.0)
    assert result.frames[0].slices == (PANEL,)


def test_metadata_changes_no_rectangle_the_packer_places():
    """Determinism is decided by rectangles alone; the pin says so from outside
    the packer."""
    plain = [_sprite("a"), _sprite("b")]
    dressed = [
        _sprite("a", SpriteMeta(pivot=(1.0, 2.0), slices=(PANEL,))),
        _sprite("b", SpriteMeta(pivot=(3.0, 4.0))),
    ]
    settings = PackSettings(power_of_two=False)
    a, b = layout(plain, settings), layout(dressed, settings)
    assert (a.width, a.height) == (b.width, b.height)
    assert [(f.key, f.x, f.y, f.w, f.h) for f in a.frames] == [
        (f.key, f.x, f.y, f.w, f.h) for f in b.frames
    ]


def test_a_rename_carries_the_metadata_through():
    """The one place it could quietly stop: the rebuild names its fields."""
    doc = PackDoc()
    source = doc.add_source(_sprite("a", SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,))))
    doc.rename_source(source.uid, "hero")
    rebuilt = doc.sprites()[0]
    assert rebuilt.name == "hero"
    assert rebuilt.meta == SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,))


# --- the TexturePacker sidecar ------------------------------------------------


def _entry(meta: SpriteMeta = EMPTY_META, **settings) -> dict:
    # MaxRects unless a case says otherwise: trimming is that mode's, since a
    # grid cell is addressed by arithmetic and cannot be re-registered.
    settings.setdefault("mode", "maxrects")
    result = layout([_sprite("a", meta)], PackSettings(power_of_two=False, **settings))
    return texturepacker.tp_json(result, image_name="atlas.png")["frames"][0]


def test_a_sprite_with_no_pivot_keeps_the_documented_centre():
    entry = _entry()
    assert entry["pivot"] == {"x": 0.5, "y": 0.5}
    assert "slices" not in entry


def test_a_pivot_is_a_fraction_of_the_trimmed_frame():
    """Exactly as ``spriteSourceSize`` is, because the sprite in the atlas is
    the trimmed one: the mark is at (2, 2)-(10, 8), so a pivot at canvas (6, 5)
    is halfway across it and half a row down."""
    entry = _entry(SpriteMeta(pivot=(6.0, 5.0)))
    assert entry["spriteSourceSize"] == {"x": 2, "y": 2, "w": 8, "h": 6}
    assert entry["pivot"] == {"x": 0.5, "y": 0.5}
    assert _entry(SpriteMeta(pivot=(2.0, 2.0)))["pivot"] == {"x": 0.0, "y": 0.0}
    assert _entry(SpriteMeta(pivot=(10.0, 8.0)))["pivot"] == {"x": 1.0, "y": 1.0}


def test_an_untrimmed_pack_normalizes_against_the_whole_canvas():
    entry = _entry(SpriteMeta(pivot=(8.0, 6.0)), trim=False)
    assert entry["spriteSourceSize"] == {"x": 0, "y": 0, "w": 16, "h": 12}
    assert entry["pivot"] == {"x": 0.5, "y": 0.5}


def test_a_fully_transparent_sprite_divides_safely():
    """It trims to 1x1 rather than to nothing, which is what makes the pivot
    division safe with no guard anywhere in the writer.

    The 2026-09-16 audit (packwright-08) found this test pinning the *wrong*
    number: normalising against that 1x1 box put a blank sprite's explicit
    pivot dozens of pixels outside its own frame. The fix normalises an empty
    sprite's pivot against its own untrimmed canvas instead, which is what
    keeps the emitted fraction between 0 and 1 -- see
    ``test_pivot_on_an_empty_sprite_with_an_explicit_pivot_stays_a_fraction_between_zero_and_one``
    for the case that first exposed it."""
    blank = Sprite(
        key="a",
        name="a",
        pixels=np.zeros((12, 16, 4), dtype=np.uint8),
        meta=SpriteMeta(pivot=(4.0, 4.0)),
    )
    result = layout([blank], PackSettings(power_of_two=False))
    assert (result.frames[0].w, result.frames[0].h) == (1, 1)
    entry = texturepacker.tp_json(result, image_name="a.png")["frames"][0]
    assert entry["pivot"] == {"x": 4.0 / 16, "y": 4.0 / 12}


def test_pivot_on_an_empty_sprite_with_an_explicit_pivot_stays_a_fraction_between_zero_and_one():
    """The 2026-09-16 audit, packwright-08: a blank "pause" frame sharing its
    neighbours' foot pivot exported that pivot as the raw untrimmed-canvas
    pixel coordinate, because ``trim_rect`` always collapses an all-transparent
    sprite's trim rectangle to a 1x1 box at (0, 0) and ``_pivot`` normalised
    against *that* box regardless of where the pivot was actually set. A 64x64
    blank frame with ``pivot=(32.0, 60.0)`` -- matched to a same-pivot solid
    neighbour -- must come out with the same fraction the solid neighbour gets,
    not the raw pixel coordinate."""
    solid = Sprite(
        key="a",
        name="a",
        pixels=np.full((64, 64, 4), 255, dtype=np.uint8),
        meta=SpriteMeta(pivot=(32.0, 60.0)),
    )
    blank = Sprite(
        key="b",
        name="b",
        pixels=np.zeros((64, 64, 4), dtype=np.uint8),
        meta=SpriteMeta(pivot=(32.0, 60.0)),
    )
    result = layout([solid, blank], PackSettings(mode="maxrects", power_of_two=False))
    payload = texturepacker.tp_json(result, image_name="atlas.png")["frames"]
    by_name = {entry["filename"]: entry for entry in payload}
    solid_pivot = by_name["a.png"]["pivot"]
    blank_pivot = by_name["b.png"]["pivot"]
    assert solid_pivot == {"x": 0.5, "y": 0.9375}
    assert blank_pivot == solid_pivot
    assert 0.0 <= blank_pivot["x"] <= 1.0
    assert 0.0 <= blank_pivot["y"] <= 1.0


def test_the_coverage_line_is_not_recomputed_between_packs():
    """The 2026-09-16 audit, packwright-08: the items pane's "-- N% covered"
    line (``packwright_items._coverage_pct``) and the preview pane's source-
    pixel-area sum (``packwright_preview._source_area``) each recomputed a
    full Python-level sum over every packed frame or every source on every
    single frame either pane draws, with no memoisation keyed on
    ``tab.pack_generation`` -- unlike ``packwright_mode.source_index``, fixed
    for the same shape by the 2026-09-07 audit's packwright-07. Proven by
    counting how many times the underlying sequence is actually iterated:
    each helper must touch it once per pack, not once per frame drawn."""
    from realmspinner.studio.modes.packwright.ui.panes import items as packwright_items
    from realmspinner.studio.modes.packwright.ui.panes import preview as packwright_preview

    class _CountingList(list):
        def __init__(self, *args) -> None:
            super().__init__(*args)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    class _Frame:
        def __init__(self, w: int, h: int) -> None:
            self.w, self.h = w, h

    class _Layout:
        def __init__(self, frames) -> None:
            self.frames = frames
            self.width = 10
            self.height = 10

    class _ItemsTab:
        def __init__(self, layout) -> None:
            self.layout = layout
            self.pack_generation = 1

    frames = _CountingList([_Frame(2, 2), _Frame(3, 3)])
    tab = _ItemsTab(_Layout(frames))
    first = packwright_items._coverage_pct(tab)
    second = packwright_items._coverage_pct(tab)
    assert first == second == 13  # (4 + 9) / 100
    assert frames.iterations == 1, "a second draw at the same pack_generation re-summed"

    # A new pack lands: the cache must follow it, not stay pinned.
    frames_two = _CountingList([_Frame(1, 1)])
    tab.layout = _Layout(frames_two)
    tab.pack_generation = 2
    third = packwright_items._coverage_pct(tab)
    assert third == 1
    assert frames_two.iterations == 1

    class _Sprite:
        def __init__(self, w: int, h: int) -> None:
            self.width, self.height = w, h

    class _Source:
        def __init__(self, w: int, h: int) -> None:
            self.sprite = _Sprite(w, h)

    class _Doc:
        def __init__(self, sources) -> None:
            self.sources = sources

    class _PreviewTab:
        def __init__(self, doc) -> None:
            self.doc = doc
            self.pack_generation = 1

    sources = _CountingList([_Source(4, 4), _Source(2, 2)])
    ptab = _PreviewTab(_Doc(sources))
    area_first = packwright_preview._source_area(ptab)
    area_second = packwright_preview._source_area(ptab)
    assert area_first == area_second == 20
    assert sources.iterations == 1, "a second draw at the same pack_generation re-summed"


def test_the_slice_block_is_in_source_image_space():
    """No trim interaction: it describes the picture the artist drew, and a
    consumer that wants the trimmed frame has ``spriteSourceSize`` to
    subtract."""
    entry = _entry(SpriteMeta(slices=(PANEL,)))
    assert entry["slices"] == [
        {
            "name": "panel",
            "bounds": {"x": 2, "y": 3, "w": 8, "h": 6},
            "pivot": {"x": 6.0, "y": 9.0},
            "center": {"x": 4, "y": 5, "w": 4, "h": 2},
        }
    ]


def test_the_schema_keeps_its_shape_when_nothing_is_carried():
    """The published TexturePacker "JSON (Array)" shape, asserted as a set so a
    ninth key appearing unconditionally would fail here rather than in whatever
    engine reads the file."""
    entry = _entry()
    assert set(entry) == {
        "filename", "frame", "rotated", "trimmed",
        "spriteSourceSize", "sourceSize", "pivot",
    }
    assert set(entry["pivot"]) == {"x", "y"}
    assert set(entry["frame"]) == {"x", "y", "w", "h"}
    assert set(entry["spriteSourceSize"]) == {"x", "y", "w", "h"}
    assert set(entry["sourceSize"]) == {"w", "h"}


def test_the_slice_block_is_the_only_key_metadata_adds():
    assert set(_entry(SpriteMeta(pivot=(1.0, 1.0), slices=(PANEL,)))) - set(_entry()) == {
        "slices"
    }


def test_an_atlas_of_plain_sprites_serializes_to_what_it_always_did():
    plain = layout([_sprite("a"), _sprite("b")], PackSettings(power_of_two=False))
    first = texturepacker.tp_bytes(plain, image_name="atlas.png")
    assert first == texturepacker.tp_bytes(plain, image_name="atlas.png")
    assert b"slices" not in first


# --- .rpack -------------------------------------------------------------------


def _packed(meta: SpriteMeta = EMPTY_META) -> PackDoc:
    doc = PackDoc()
    doc.add_source(_sprite("a", meta))
    doc.mark_saved()
    return doc


def test_a_document_with_no_metadata_writes_the_manifest_it_always_wrote():
    entry = json.loads(rpack.manifest_json(_packed()))["sources"][0]
    assert set(entry) == {"key", "name", "name_override", "image"}


def test_metadata_round_trips_through_the_file():
    meta = SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,))
    back = rpack.read_rpack(rpack.rpack_bytes(_packed(meta)))
    assert back.sources[0].sprite.meta == meta


def test_a_slice_with_no_pivot_or_centre_round_trips_as_none():
    meta = SpriteMeta(slices=(SliceSpec(name="hit", x=1, y=2, w=3, h=4),))
    back = rpack.read_rpack(rpack.rpack_bytes(_packed(meta)))
    assert back.sources[0].sprite.meta == meta


def test_two_saves_of_a_document_with_metadata_are_byte_identical():
    doc = _packed(SpriteMeta(pivot=(6.0, 9.0), slices=(PANEL,)))
    assert rpack.rpack_bytes(doc) == rpack.rpack_bytes(doc)
    back = rpack.read_rpack(rpack.rpack_bytes(doc))
    assert rpack.rpack_bytes(back) == rpack.rpack_bytes(doc)


def test_a_manifest_from_before_the_keys_reads_clean():
    raw = rpack.rpack_bytes(_packed())
    assert rpack.read_rpack(raw).sources[0].sprite.meta == EMPTY_META


@pytest.mark.parametrize(
    "meta",
    [
        {"pivot": "middle"},
        {"pivot": {"x": 1.0}},
        {"slices": [{"name": "x"}]},
        {"slices": [{"name": "x", "bounds": {"x": 0, "y": 0, "w": 2}}]},
        {"slices": "some"},
    ],
)
def test_a_malformed_metadata_field_is_refused_by_name(meta: dict):
    """``.rpack`` is ours and versioned, so it recognises or refuses -- and the
    refusal names the source, because "malformed" alone cannot be acted on in a
    manifest with forty sprites in it."""
    with pytest.raises(ValueError, match="metadata for 'a'"):
        rpack._meta_from({"key": "a", **meta}, "a")


def test_a_refusal_reaches_the_reader_rather_than_being_swallowed():
    doc = _packed(SpriteMeta(pivot=(6.0, 9.0)))
    raw = rpack.rpack_bytes(doc)
    broken = _rewrite(raw, lambda m: _break_pivot(m))
    with pytest.raises(ValueError, match="metadata for 'a'"):
        rpack.read_rpack(broken)


def _break_pivot(manifest: dict) -> dict:
    manifest["sources"][0]["pivot"] = "middle"
    return manifest


def _rewrite(raw: bytes, edit) -> bytes:
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == rpack.MANIFEST:
                data = json.dumps(edit(json.loads(data))).encode()
            dst.writestr(info, data, info.compress_type)
    return out.getvalue()
