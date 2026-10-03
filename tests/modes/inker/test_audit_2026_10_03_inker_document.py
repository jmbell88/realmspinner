"""Regressions for the 2026-10-03 audit's inker document findings (inker-02,
03, 11, 12, 13, 14, 15)."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.pixel.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
HOLE = (0, 0, 0, 0)


def test_redoing_an_added_layer_after_undoing_a_later_crop_restores_its_original_pixels():
    # inker-02: the add edit holds the live Layer, crop rebinds its pixels in
    # place, and undoing the crop swapped the stack for copies -- leaving the
    # held layer cropped, so the redo of the add installed a 4x4 layer on an
    # 8x8 canvas ("a layer is canvas-sized").
    doc = Document.blank(8, 8)
    doc.add_layer()
    assert doc.crop((0, 0, 4, 4))
    doc.undo()  # crop
    doc.undo()  # add
    doc.redo()  # add
    assert len(doc.stack) == 2
    assert doc.size == (8, 8)
    assert all(layer.size == (8, 8) for layer in doc.stack)
    doc.redo()  # crop replays on the redone layer too
    assert doc.size == (4, 4)
    assert all(layer.size == (4, 4) for layer in doc.stack)


def test_redoing_an_autovivified_cel_after_undoing_a_later_crop_keeps_the_canvas_size():
    # inker-02, the animated half: the redone cel silently resized the
    # document. A stroke on an empty cel autovivifies it through CelSetEdit.
    doc = Document.blank(8, 8)
    doc.add_frame()  # animates; frame 1's cel is the empty placeholder
    doc.set_current_frame(1)
    assert doc.apply_pixels(
        doc.stack[0].uid, (1, 1, 2, 2), np.full((1, 1, 4), 255, dtype=np.uint8)
    )
    assert doc.crop((0, 0, 4, 4))
    assert doc.undo()  # crop
    assert doc.undo()  # the autovivifying write
    assert doc.redo()  # the write again, before the crop is redone
    assert doc.size == (8, 8)
    for layer in doc.anim.unique_cel_layers():
        assert layer.size == (8, 8)
    assert doc.redo()  # crop
    assert doc.size == (4, 4)


def test_redoing_a_slice_add_after_undoing_a_rotate_keeps_the_original_bounds():
    # inker-13: _remap mutates the held Slice in place; the undo swapped in
    # copies, so a redo of the add re-inserted the already-rotated rectangle.
    doc = Document.blank(8, 8)
    uid = doc.add_slice((0, 0, 2, 2), name="s").uid
    doc.rotate90(1)
    doc.undo()  # rotate
    doc.undo()  # add slice
    doc.redo()  # add slice
    (got,) = doc.slices
    assert got.bounds == (0, 0, 2, 2)
    doc.redo()  # rotate
    doc.undo()
    assert [s.bounds for s in doc.slices] == [(0, 0, 2, 2)]
    assert uid == doc.slices[0].uid


def test_moving_an_indexed_layer_vacates_to_the_transparent_index_and_undo_restores_it():
    # inker-03: _translated zero-filled the vacated index strip with slot 0,
    # an opaque colour here (transparent index is 1), and the strip lay outside
    # the pixel-measured patch box so undo never put it back.
    doc = Document.blank(4, 1)
    doc.stack[0].pixels[0] = [HOLE, BLUE, BLUE, HOLE]
    doc.invalidate_all()
    doc.convert_to_indexed([RED, HOLE, BLUE], "nearest", transparent=1)
    assert doc.stack[0].indices.tolist() == [[1, 2, 2, 1]]
    assert doc.begin_layer_move()
    assert doc.preview_layer_move(1, 0)
    assert doc.commit_layer_move()
    doc.check_materialized()
    assert doc.stack[0].indices.tolist() == [[1, 1, 2, 2]]
    assert doc.undo()
    doc.check_materialized()
    assert doc.stack[0].indices.tolist() == [[1, 2, 2, 1]]


def test_despeckle_with_a_fractional_speck_below_one_half_does_not_crash_and_changes_nothing(
    monkeypatch,
):
    # inker-04: speck 0.3 rounds to a window of 1 and Pillow's MedianFilter(1)
    # kills the process natively. The spy turns that into a test failure
    # instead of taking pytest down with it.
    from PIL import ImageFilter

    from realmspinner.kernels.pixel import filters

    def guarded(size):
        assert size >= 3, f"MedianFilter({size}) is a native crash in Pillow 12.3"
        return real(size)

    real = ImageFilter.MedianFilter
    monkeypatch.setattr(ImageFilter, "MedianFilter", guarded)
    pixels = np.random.default_rng(3).integers(0, 256, (6, 6, 4), dtype=np.uint8)
    for speck in (0.01, 0.3, 0.5):
        got = filters.despeckle(pixels, speck=speck)
        assert np.array_equal(got, pixels)
        assert got is not pixels


def test_to_background_on_an_empty_animated_cel_is_undoable_after_a_later_stroke_is_undone():
    # inker-11: the cel to_background autovivified stayed queued, the next
    # stroke's step swallowed it, and undoing that stroke removed the
    # background's cel -- after which undoing the conversion raised.
    doc = Document.blank(8, 8)
    doc.add_frame()
    assert doc.anim.is_placeholder(doc.stack[0])
    assert doc.to_background() is True
    assert doc._pending_cels == []
    uid = doc.stack[0].uid
    assert doc.apply_pixels(uid, (1, 1, 2, 2), np.full((1, 1, 4), 200, dtype=np.uint8))
    assert doc.undo()  # the stroke: the background cel must survive it
    assert not doc.anim.is_placeholder(doc.stack[0])
    assert doc.stack[0].background is True
    assert doc.undo()  # the conversion itself: used to raise "read-only"
    assert doc.stack[0].background is False
    assert doc.redo()
    assert doc.stack[0].background is True


def test_a_refused_apply_pixels_does_not_leave_an_autovivified_cel_queued():
    # inker-11, the apply_pixels half: shape-mismatch and off-canvas exits.
    doc = Document.blank(8, 8)
    doc.add_frame()
    uid = doc.stack[0].uid
    assert doc.anim.is_placeholder(doc.stack[0])
    assert not doc.apply_pixels(uid, (0, 0, 2, 2), np.zeros((1, 1, 4), dtype=np.uint8))
    assert not doc.apply_pixels(uid, (20, 20, 22, 22), np.zeros((2, 2, 4), dtype=np.uint8))
    assert doc._pending_cels == []
    assert doc.anim.is_placeholder(doc.stack[0])


def test_undoing_set_reference_on_another_frame_clears_the_tracks_flag():
    # inker-12: LayerFlagEdit addressed the frame-local layer uid, so after the
    # playhead moved an undo found no row, did nothing, and moved the head.
    doc = Document.blank(8, 8)
    doc.add_frame()
    doc.set_current_frame(0)
    assert doc.set_reference(0, True)
    track = doc.anim.tracks[0]
    assert track.reference is True
    doc.set_current_frame(1)
    assert doc.undo()
    assert track.reference is False
    assert doc.stack[0].reference is False
    doc.set_current_frame(0)
    assert doc.stack[0].reference is False
    doc.set_current_frame(1)
    assert doc.redo()
    assert track.reference is True
    assert doc.stack[0].reference is True


def test_a_stroke_that_snaps_back_to_the_old_colour_leaves_the_composite_matching_the_layers():
    # inker-15: the palette-constrained branch returned on before == after
    # without recompositing, leaving the raw dab (purple) in the composite.
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.set_palette([RED, BLUE])
    layer = doc.stack[0]
    before = layer.pixels[0:1, 0:1].copy()
    layer.pixels[0, 0] = (179, 0, 77, 255)  # the tool's raw low-opacity dab
    doc.invalidate((0, 0, 1, 1), layer_uid=layer.uid)
    assert tuple(doc.flatten()[0, 0]) == (179, 0, 77, 255)  # on screen, pre-commit
    head = doc.history.head
    doc._commit_patch(layer, (0, 0, 1, 1), before)
    assert tuple(layer.pixels[0, 0]) == RED
    assert doc.history.head == head, "nothing changed, nothing is pushed"
    assert tuple(doc.flatten()[0, 0]) == RED


def test_recolouring_a_slot_on_a_frame_with_its_own_palette_changes_that_frames_table_only():
    # inker-14: recolour_slot rewrote the document table, so the frame that
    # had its own table kept its colours and every other frame changed.
    from tests.modes.inker.test_frame_palettes import GREEN, _indexed

    doc = _indexed(frames=2)
    assert doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(1)
    assert doc.recolour_slot(1, GREEN)
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, GREEN]
    assert doc.palette_for(doc.anim.frames[0]) == [HOLE, RED]
    assert doc.palette == [HOLE, RED]
    assert doc.undo()
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, BLUE]
    # A frame with no override still edits the document's table.
    doc.set_current_frame(0)
    assert doc.recolour_slot(1, GREEN)
    assert doc.palette == [HOLE, GREEN]
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, BLUE]
