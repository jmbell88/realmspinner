"""Regression tests for the 2026-09-26 audit's w2f1 fixer slice (kernels/pixel).

Six findings: unclamped/non-finite opacity read from an ``.ora`` (inker-codecs-05),
an XML-illegal character in a layer/group/track name surviving into
``stack.xml`` (inker-codecs-06), an oversized ``palette.gpl`` member refusing
the whole archive instead of costing the constraint (inker-codecs-08),
``shift_selected`` pushing two undo steps instead of one (inker-document-03),
``merge_down``/``_merge_tracks`` baking a hidden group's contents into the
canvas (inker-document-05), and several sheet/range/tile write doors never
consulting ``write_locked`` (inker-document-06).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel import gpl, ora
from realmspinner.kernels.pixel.tiles import strip


def _rewrite_member(path: Path, member: str, data: bytes) -> None:
    """Replace one member's bytes, keeping every other member and the order.

    ``tests/modes/inker/test_ora_degrade.py``'s own helper, copied rather than
    imported: that module is not in this fixer's owned set and importing a
    private helper across test files couples two suites that should stay free
    to diverge.
    """
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        items = {name: zf.read(name) for name in names}
    items[member] = data
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, items[name])


# --- inker-codecs-05: opacity clamped on read --------------------------------


def test_clamp_opacity_forces_every_bad_value_into_range():
    """The 2026-09-26 audit, finding inker-codecs-05: a layer/group/track
    opacity read from a file went straight from the raw string to the field
    with no check at all -- ``nan``, ``inf``, a negative number and anything
    past 1.0 all landed verbatim."""
    assert ora._clamp_opacity("nan") == 1.0
    assert ora._clamp_opacity(float("inf")) == 1.0
    assert ora._clamp_opacity(float("-inf")) == 1.0
    assert ora._clamp_opacity(-5.0) == 0.0
    assert ora._clamp_opacity(5.0) == 1.0
    assert ora._clamp_opacity(0.25) == 0.25
    assert ora._clamp_opacity("not-a-number") == 1.0


def test_an_out_of_range_or_nonfinite_ora_opacity_is_clamped_on_read(tmp_path: Path):
    """The round trip: a hand-crafted ``stack.xml`` naming ``opacity="nan"``
    on a layer must not poison the document that opens -- it must clamp,
    matching every other numeric guard this reader already has."""
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[0:2, 0:2] = (5, 6, 7, 255)
    doc.invalidate_all()
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        stack_xml = zf.read("stack.xml").decode("utf-8")
    assert 'opacity="1.000000"' in stack_xml, "sanity: this is the attribute we are corrupting"
    corrupted = stack_xml.replace('opacity="1.000000"', 'opacity="nan"', 1)
    _rewrite_member(path, "stack.xml", corrupted.encode("utf-8"))

    back = ora.read_ora(path)  # must not raise, and must not carry nan through
    assert back.stack[0].opacity == 1.0


# --- inker-codecs-06: an XML-illegal name round-trips ------------------------


def test_a_layer_name_with_a_control_character_still_round_trips_through_ora(
    tmp_path: Path,
):
    """The 2026-09-26 audit, finding inker-codecs-06: a layer name carrying an
    XML-illegal control character (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F) was
    written verbatim into ``stack.xml``, producing a file this very reader
    refuses on the next open."""
    doc = inker.Document.blank(8, 8)
    doc.set_layer_props(0, name="bad\x00name\x0bhere")
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)  # must not write an unparseable stack.xml

    with zipfile.ZipFile(path) as zf:
        stack_xml = zf.read("stack.xml")
    assert b"\x00" not in stack_xml and b"\x0b" not in stack_xml

    back = ora.read_ora(path)  # must not raise
    assert "\x00" not in back.stack[0].name and "\x0b" not in back.stack[0].name


# --- inker-codecs-08: an oversized palette.gpl costs the constraint ----------


def test_an_oversized_palette_gpl_member_costs_the_constraint_not_the_file(
    tmp_path: Path,
):
    """The 2026-09-26 audit, finding inker-codecs-08: ``gpl.parse`` only
    enforces its own 65536-row ceiling, far above the document's own 256-
    colour ceiling, so a member between the two sizes parsed clean inside
    ``_read_palette``'s own try and then raised out of ``doc.set_palette`` a
    dozen lines below it, refusing the whole archive over one oversized
    member."""
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[0:2, 0:2] = (1, 2, 3, 255)
    doc.invalidate_all()
    assert doc.set_palette([(0, 0, 0, 255), (255, 255, 255, 255)])
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)

    oversized = gpl.dumps([(i % 256, 0, 0, 255) for i in range(300)])
    _rewrite_member(path, ora.PALETTE_MEMBER, oversized.encode("utf-8"))

    back = ora.read_ora(path)  # must not raise
    assert back.palette is None, "an oversized palette degrades to no constraint"


# --- inker-document-03: shift_selected is one undo step ----------------------


def test_shift_selected_is_exactly_one_undo_step_and_one_undo_restores_the_pixels():
    """The 2026-09-26 audit, finding inker-document-03: ``shift_selected`` is
    ``lift()`` followed by ``commit_floating()``, each pushing its own undo
    step, while the docstring and the "Shift pixels..." hint both promise one
    -- so one Ctrl+Z left the pixels moved and only a second one put them
    back."""
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[2:4, 2:4] = (200, 10, 10, 255)
    doc.invalidate_all()
    doc.history.clear()
    before_pixels = doc.stack.active.pixels.copy()

    doc.select_all()
    depth_before = len(doc.history._done)
    assert doc.shift_selected(1, 0)
    pushed = len(doc.history._done) - depth_before
    assert pushed == 1, f"shift_selected pushed {pushed} steps, not one"

    assert doc.history.undo(doc)
    assert np.array_equal(doc.stack.active.pixels, before_pixels), (
        "one undo must restore every pixel shift_selected moved"
    )


# --- inker-document-05: merge_down out of a hidden group ---------------------


def test_merge_down_out_of_a_hidden_group_does_not_change_the_composite():
    """The 2026-09-26 audit, finding inker-document-05: ``merge_down`` blended
    the upper row with its own ``visible``/``opacity`` only, never the fold
    from an ancestor group -- so merging a layer out of a hidden folder baked
    its pixels into the lower layer even though the canvas never showed them."""
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[...] = (10, 10, 10, 255)  # lower, layer 0
    doc.invalidate_all()
    doc.add_layer()  # upper, layer 1, now active
    doc.stack.active.pixels[...] = (250, 0, 0, 255)  # loud, opaque upper
    doc.invalidate_all()

    node = doc.group_layers([1])
    assert node is not None
    assert doc.set_group_props(node.uid, visible=False)

    before = doc.flatten(matte=False).copy()
    assert np.all(before[..., 0] == 10), "sanity: the hidden upper layer must not show yet"

    assert doc.merge_down(1)

    after = doc.flatten(matte=False)
    assert np.array_equal(before, after), (
        "merging a layer out of a hidden group must not change what the "
        "canvas shows"
    )
    assert np.all(doc.stack.active.pixels[..., 0] == 10), (
        "the hidden upper layer's loud pixels must not be baked into the lower layer"
    )


# --- inker-document-06: write_locked doors -----------------------------------


def test_sheet_verbs_paste_cels_and_place_tiles_refuse_a_write_locked_row():
    """The 2026-09-26 audit, finding inker-document-06: ``map_frames`` (and
    every sheet verb that resolves to it), ``paste_cels`` and ``place_tiles``
    never consulted ``write_locked``, so a content-locked track (or one inside
    a locked group) still took a sheet correction, a pasted cel or a placed
    tile the same as any unlocked row."""
    # -- map_frames / replace_colour_frames, through the one sheet funnel --
    doc = inker.Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    track0_uid = doc.anim.tracks[0].uid
    frame0_uid = doc.anim.frames[0].uid
    assert doc.set_layer_props(0, locked=True)
    before_cel = doc.anim.cels.get((track0_uid, frame0_uid))
    before_pixels = None if before_cel is None else before_cel.pixels.copy()

    changed = doc.map_frames(track0_uid, [0], lambda before: np.zeros_like(before))
    assert changed is False
    after_cel = doc.anim.cels.get((track0_uid, frame0_uid))
    if before_pixels is not None and after_cel is not None:
        assert np.array_equal(after_cel.pixels, before_pixels)

    assert (
        doc.replace_colour_frames(
            track0_uid, [0], (0, 0, 0, 0), (255, 255, 255, 255)
        )
        is False
    )

    # -- paste_cels: a locked target track must refuse the whole paste -----
    doc.add_layer()  # track 1, unlocked, now active
    doc.select_all()
    assert doc.fill_selection((10, 20, 30, 255))
    clip = doc.copy_cels(1, 1, 0, 0)
    assert clip is not None
    before_target = doc.anim.cels.get((track0_uid, frame0_uid))
    assert doc.paste_cels(clip, 0, 0) is False
    assert doc.anim.cels.get((track0_uid, frame0_uid)) is before_target

    # -- place_tiles: a locked tilemap layer must refuse the write ----------
    tdoc = inker.Document.blank(16, 16)
    tile = np.zeros((16, 16, 4), dtype=np.uint8)
    tile[..., 3] = 255
    tileset_stack = np.stack([np.zeros((16, 16, 4), dtype=np.uint8), tile], axis=0)
    slot = tdoc.add_tileset(strip(tileset_stack))
    cel = tdoc.add_tilemap_layer(slot.uid)
    index = tdoc.stack.index_of(cel.uid)
    assert tdoc.set_layer_props(index, locked=True)
    before_refs = cel.refs.copy()
    patch = np.ones((1, 1), dtype=np.uint32)
    assert tdoc.place_tiles(cel.uid, (0, 0), patch) is False
    assert np.array_equal(tdoc.layer_by_uid(cel.uid).refs, before_refs)
