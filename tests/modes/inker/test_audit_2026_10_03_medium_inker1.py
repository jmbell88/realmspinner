"""Regressions for the 2026-10-03 audit's Medium inker findings 26-33.

Codecs (26-28): a file this build can open must not hold a time bomb for the
frame thread, and a document it can save must reopen. Document (29-33): the
palette ops, the stroke session, the content lock, the property writers and the
group history each had one door the neighbouring doors had been taught to guard.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.pixel import ora
from realmspinner.kernels.pixel.document import Document

HOLE = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def _rewrite_member(path: Path, name: str, data: bytes) -> None:
    with zipfile.ZipFile(path) as zf:
        members = [(info, zf.read(info.filename)) for info in zf.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for info, body in members:
            out.writestr(info, data if info.filename == name else body)


def _patch_animation(path: Path, edit) -> None:
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.ANIMATION_MEMBER))
    edit(payload)
    _rewrite_member(path, ora.ANIMATION_MEMBER, json.dumps(payload).encode("utf-8"))


def _indexed_animation(frames: int = 2) -> Document:
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED], "nearest", transparent=0)
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


# --- inker-26 ---------------------------------------------------------------


def test_a_per_frame_palette_entry_past_a_byte_is_clamped_on_read(tmp_path: Path):
    doc = _indexed_animation()
    doc.set_frame_palette([HOLE, BLUE], 1)
    path = tmp_path / "cycle.ora"
    ora.write_ora(doc, path)

    def corrupt(payload: dict) -> None:
        payload["frames"][1]["palette"] = [[0, 0, 0, 0], [300, -5, 0, 255]]

    _patch_animation(path, corrupt)
    back = Document.load(path)
    assert back.anim is not None, "the timeline survives; only the bad channel is clamped"
    table = back.anim.frame_palettes[back.anim.frames[1].uid]
    assert table[1] == (255, 0, 0, 255)
    back.set_current_frame(1)
    back.stack.flatten()  # raised OverflowError out of the uint8 lut before the fix


# --- inker-27 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["￾", "￿", "\ud800"], ids=["fffe", "ffff", "surrogate"])
def test_a_name_with_an_xml_noncharacter_still_reopens(tmp_path: Path, bad: str):
    doc = Document.blank(4, 4)
    doc.set_layer_props(0, name=f"a{bad}b")
    doc.add_layer(f"c{bad}d")
    doc.group_layers([0, 1], name=f"g{bad}h")
    path = tmp_path / "n.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)  # raised "stack.xml is not a readable XML document"
    assert [layer.name for layer in back.stack] == ["ab", "cd"]


def test_an_animation_track_name_with_an_xml_noncharacter_still_reopens(tmp_path: Path):
    doc = _indexed_animation()
    doc.set_layer_props(0, name="t￿x")
    path = tmp_path / "n.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)  # stack.xml carries a stripped name; animation.json the original
    assert back.anim is not None and len(back.anim.tracks) == 1


# --- inker-28 ---------------------------------------------------------------


def test_an_animation_json_with_too_many_tags_falls_back_to_the_flat_read(
    tmp_path: Path, monkeypatch
):
    doc = _indexed_animation()
    path = tmp_path / "t.ora"
    ora.write_ora(doc, path)
    monkeypatch.setattr(ora, "MAX_ORA_METADATA_ENTRIES", 3)
    _patch_animation(
        path,
        lambda payload: payload.__setitem__(
            "tags", [{"name": f"t{i}", "start": 0, "end": 0} for i in range(4)]
        ),
    )
    back = ora.read_ora(path)
    assert back.anim is None, "the whole timeline degrades, as every sibling list does"
    # ...and the same list at the ceiling still opens.
    _patch_animation(
        path,
        lambda payload: payload.__setitem__(
            "tags", [{"name": f"t{i}", "start": 0, "end": 0} for i in range(3)]
        ),
    )
    assert len(ora.read_ora(path).anim.tags) == 3


# --- inker-29 ---------------------------------------------------------------


def _frame_one_pixel(doc: Document) -> tuple:
    doc.set_current_frame(1)
    out = tuple(np.round(doc.stack.composite_region((0, 0, 1, 1))[0, 0] * 255))
    doc.set_current_frame(0)
    return out


def _three_slot_animation() -> Document:
    """Slot 1 red, slot 2 green; frame 1 overrides slot 1 to blue."""
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED, GREEN], "nearest", transparent=0)
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    assert doc.set_frame_palette([HOLE, BLUE, GREEN], 1)
    doc.history.clear()  # the walk-back loops below must stop at the setup
    return doc


def test_reordering_the_palette_keeps_a_frames_own_table_aligned_with_the_planes():
    doc = _three_slot_animation()
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.move_slot(1, 2)
    # The pixels moved with their slot (1 -> 2), so each frame still shows what
    # it showed: frame 1 blue, not green.
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    doc.set_current_frame(0)
    assert tuple(np.round(doc.stack.composite_region((0, 0, 1, 1))[0, 0] * 255)) == (
        255.0, 0.0, 0.0, 255.0,
    )
    assert doc.undo()
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.anim.frame_palettes[doc.anim.frames[1].uid] == [HOLE, BLUE, GREEN]
    assert doc.redo()
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)


def test_sorting_inserting_and_removing_slots_keep_a_frames_own_table_aligned():
    doc = _three_slot_animation()
    # insert_ramp widens the table: every slot above the insertion shifts.
    assert doc.insert_ramp(0, 1, 1)
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.undo()
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.redo()
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.undo()

    # sort_palette: any reorder that moves slot 1.
    assert doc.sort_palette("red", descending=True)
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    while doc.history.can_undo:
        doc.undo()

    # remove_slot merges slot 2 away; slot 1 keeps meaning blue on frame 1.
    assert doc.remove_slot(2)
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)
    assert doc.anim.frame_palettes[doc.anim.frames[1].uid] == [HOLE, BLUE]
    assert doc.undo()
    assert doc.anim.frame_palettes[doc.anim.frames[1].uid] == [HOLE, BLUE, GREEN]
    assert _frame_one_pixel(doc) == (0.0, 0.0, 255.0, 255.0)


# --- inker-30 ---------------------------------------------------------------


def test_changing_frame_mid_stroke_still_records_the_paint_as_one_undo_step():
    doc = Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    doc.set_current_frame(0)
    doc.history.clear()
    assert doc.begin_stroke((1, 1), colour=(200, 10, 10, 255), size=1)
    doc.stroke_to((4, 4))
    doc.set_current_frame(1)  # Home / End are not refused during a drag
    doc.stroke_to((5, 5))
    assert doc.end_stroke() is True
    assert doc.history.can_undo, "the dabs on frame 0's cel must be undoable"
    doc.set_current_frame(0)
    assert doc.stack.active.pixels[1, 1, 3] == 255
    assert doc.undo()
    assert doc.stack.active.pixels[1, 1, 3] == 0
    assert not doc.history.can_undo, "one stroke, one step"


# --- inker-31 ---------------------------------------------------------------


def _animated_with_cel_on_frame_zero():
    doc = Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    doc.set_current_frame(0)
    assert doc.begin_stroke((1, 1), colour=(200, 10, 10, 255), size=1)
    doc.stroke_to((2, 2))
    doc.end_stroke()
    uid = doc.stack.active.uid
    return doc, uid


def test_apply_pixels_refuses_a_cel_whose_track_was_locked_after_the_playhead_left_its_frame():
    doc, uid = _animated_with_cel_on_frame_zero()
    doc.set_current_frame(1)
    assert doc.set_layer_props(0, locked=True)  # the *track*, from frame 1
    patch = np.full((2, 2, 4), 255, dtype=np.uint8)
    assert doc.apply_pixels(uid, (4, 4, 6, 6), patch) is False
    doc.set_current_frame(0)
    assert doc.stack.active.pixels[4, 4, 3] == 0, "the locked track took no pixels"


def test_apply_pixels_refuses_an_off_frame_cel_inside_a_locked_group():
    doc = Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    doc.add_layer()  # a second track, so the cel's uid is not its track's uid
    doc.set_current_frame(0)
    assert doc.begin_stroke((1, 1), colour=(200, 10, 10, 255), size=1)
    doc.stroke_to((2, 2))
    doc.end_stroke()
    uid = doc.stack.active.uid
    track = doc.anim.tracks[doc.stack.active_index]
    assert uid != track.uid
    node = doc.group_layers([doc.stack.active_index])
    assert node is not None
    doc.set_current_frame(1)
    assert doc.set_group_props(node.uid, locked=True)
    patch = np.full((2, 2, 4), 255, dtype=np.uint8)
    assert doc.apply_pixels(uid, (4, 4, 6, 6), patch) is False


def test_apply_pixels_still_lands_on_an_unlocked_off_frame_cel():
    doc, uid = _animated_with_cel_on_frame_zero()
    doc.set_current_frame(1)
    patch = np.full((2, 2, 4), 255, dtype=np.uint8)
    assert doc.apply_pixels(uid, (4, 4, 6, 6), patch) is True


# --- inker-32 ---------------------------------------------------------------


def test_set_layers_props_and_set_tracks_props_refuse_an_unknown_blend_before_pushing():
    still = Document.blank(4, 4)
    still.add_layer()
    still.history.clear()
    with pytest.raises(ValueError, match="blend"):
        still.set_layers_props(None, blend="bogus-mode")
    with pytest.raises(ValueError, match="blend"):
        still._set_row_props({0: {"blend": "bogus-mode"}})
    assert all(layer.blend == "normal" for layer in still.stack)
    assert not still.history.can_undo
    still.invalidate_all()  # would raise from the composite had the value landed

    anim = Document.blank(4, 4)
    anim.ensure_animation()
    anim.add_frame()
    anim.history.clear()
    with pytest.raises(ValueError, match="blend"):
        anim.set_tracks_props([0], blend="bogus-mode")
    with pytest.raises(ValueError, match="blend"):
        anim.set_range_props(0, 0, blend="bogus-mode")
    with pytest.raises(ValueError, match="blend"):
        anim.set_layers_props(None, blend="bogus-mode")
    assert anim.anim.tracks[0].blend == "normal"
    assert not anim.history.can_undo
    anim.invalidate_all()


# --- inker-33 ---------------------------------------------------------------


def test_redoing_a_group_add_after_undoing_a_rotate_restores_the_node_as_it_was_recorded():
    doc = Document.blank(8, 6)
    doc.add_layer()
    node = doc.group_layers([0, 1])
    guid = node.uid
    assert doc.set_group_props(guid, visible=False)
    doc.rotate90(1)
    assert doc.undo()  # the whole-canvas op
    assert doc.undo()  # hide
    assert doc.undo()  # group add
    assert guid not in doc.groups
    assert doc.redo()  # group add only
    assert doc.groups[guid].visible is True, "the node carried state from still-undone steps"
    assert doc.redo()  # hide
    assert doc.groups[guid].visible is False


def test_undoing_a_whole_canvas_op_does_not_let_later_group_edits_rewrite_its_snapshot():
    doc = Document.blank(8, 6)
    doc.add_layer()
    node = doc.group_layers([0, 1])
    guid = node.uid
    doc.rotate90(1)
    assert doc.undo()
    # A write that is not a history step (what the next group edit does to the
    # live node), so the redo below survives it.
    doc._set_group_props(guid, {"visible": False})
    assert doc.redo()  # the rotate again
    assert doc.undo()  # restores the snapshot taken *before* that write
    assert doc.groups[guid].visible is True
