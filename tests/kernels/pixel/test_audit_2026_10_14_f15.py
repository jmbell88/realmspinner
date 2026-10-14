"""Regression tests for the 2026-10-14 audit's F15 batch: the Inker document and
Flourish findings whose files live under ``kernels/pixel``.

inker-22 (a duplicated frame's side tables), inker-12 (Flourish cels in a
non-RGB document), inker-13 (the other look's tracks), inker-25 (two effects'
tags), inker-26 and inker-27 (the conflict memo, the frame rate), inker-10 and
inker-23 (indexed re-conversion and remap), inker-08 (flatten and redo),
inker-09 (merge down), inker-24 (convert preview stamps), inker-38 (the
timeline's link query).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import time

import numpy as np
import pytest

from realmspinner.core.undo import Edit, UndoStack
from realmspinner.kernels.pixel import _doc_flourish
from realmspinner.kernels.pixel.animation import Note
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import presets

HOLE = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


def _small(name: str = "explosion", **over):
    rec = presets.load(name)
    return dataclasses.replace(rec, width=24, height=24, supersample=1, **over)


# -- inker-22: duplicating a frame --------------------------------------------------------------


def _doc_with_side_tables() -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.add_layer("b")
    doc.stack[1].pixels[:, :] = GREEN
    doc.invalidate_all()
    doc.ensure_animation()
    assert doc.set_cel_opacity(0.25, track_index=1, frame_index=0)
    assert doc.set_cel_z(-1, track_index=1, frame_index=0)
    assert doc.set_cel_note(Note(text="hit"), track_index=1, frame_index=0)
    return doc


def _slot(doc: Document, frame_index: int):
    track = doc.anim.tracks[1].uid
    frame = doc.anim.frames[frame_index].uid
    return (
        doc.anim.cel_alpha(track, frame),
        doc.anim.cel_zindex(track, frame),
        doc.anim.cel_note(track, frame).text,
    )


@pytest.mark.parametrize("how", ["copy", "link", "range"])
def test_duplicating_a_frame_carries_each_cels_opacity_z_and_note(how):
    doc = _doc_with_side_tables()
    assert _slot(doc, 0) == (0.25, -1, "hit")
    if how == "range":
        assert doc.duplicate_range(0, 0)
    else:
        doc.add_frame(**{how: True})
    assert _slot(doc, 1) == (0.25, -1, "hit")
    # The original keeps what it had, and the copy survives an undo and a redo
    # (the tables are keyed on a uid the step re-puts, so redo finds them).
    assert _slot(doc, 0) == (0.25, -1, "hit")
    doc.undo()
    doc.redo()
    assert _slot(doc, 1) == (0.25, -1, "hit")


def test_duplicating_a_frame_carries_the_frames_own_colour_table():
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED], "nearest", transparent=0)
    doc.ensure_animation()
    doc.set_frame_palette([HOLE, BLUE], 0)
    doc.add_frame(copy=True)
    first, second = (doc.anim.frames[i].uid for i in (0, 1))
    assert doc.anim.frame_palette(second) == doc.anim.frame_palette(first)
    assert doc.frame_flat(second)[0, 0].tolist() == doc.frame_flat(first)[0, 0].tolist()


# -- inker-12: Flourish cels in a non-RGB document ------------------------------------------------


def _effect_cels(doc: Document, group: int):
    state = doc.flourish_state(group)
    owned = set(state.tracks.values())
    return [cel for (track, _f), cel in doc.anim.cels.items() if track in owned]


def test_flourish_cels_minted_in_an_indexed_document_carry_a_plane_on_the_palette():
    doc = Document.blank(24, 24)
    table = [HOLE, (255, 255, 255, 255), RED, (255, 200, 0, 255)]
    doc.convert_to_indexed(table, transparent=0)
    group = doc.insert_flourish(B.bake(_small()))
    cels = _effect_cels(doc, group)
    assert cels
    assert all(cel.indices is not None for cel in cels)
    allowed = {tuple(c) for c in table}
    for cel in cels:
        visible = cel.pixels[cel.pixels[..., 3] > 0]
        assert {tuple(int(v) for v in p) for p in visible} <= allowed
    # ... and they are the document's slots, so a recolour reaches them.
    assert any((cel.indices == 2).any() for cel in cels), "the fixture must use slot 2"
    assert doc.recolour_slot(2, GREEN)
    for cel in cels:
        assert (cel.pixels[cel.indices == 2] == GREEN).all()


def test_an_effect_inserted_into_an_indexed_document_holds_only_palette_colours_and_index_planes():
    doc = Document.blank(24, 24)
    table = [HOLE, (255, 255, 255, 255), RED]
    doc.convert_to_indexed(table, transparent=0)
    group = doc.insert_flourish(B.bake(_small()))
    # A regenerate that adds a track mints through the same door.
    rec = doc.flourish_state(group).recipe
    doc.apply_flourish(group, B.bake(dataclasses.replace(rec, seed=rec.seed + 1)), force=True)
    for cel in _effect_cels(doc, group):
        assert cel.indices is not None
        assert np.array_equal(
            cel.pixels[cel.indices == 0], np.zeros_like(cel.pixels[cel.indices == 0])
        )
    # A snapshot track from the image model too.
    cels = {0: np.full((24, 24, 4), (10, 200, 30, 255), np.uint8)}
    uid = doc.insert_flourish_track(group, "restyled", cels)
    painted = doc.anim.cels[(uid, doc.anim.frames[0].uid)]
    assert painted.indices is not None
    assert {tuple(int(v) for v in p) for p in painted.pixels.reshape(-1, 4)} <= {
        tuple(c) for c in table
    }


def test_flourish_cels_in_a_grayscale_document_are_grey_and_regenerate_without_ghost_conflicts():
    doc = Document.blank(24, 24)
    doc.convert_to_grayscale()
    group = doc.insert_flourish(B.bake(_small()))
    for cel in _effect_cels(doc, group):
        px = cel.pixels[cel.pixels[..., 3] > 0]
        assert ((px[:, 0] == px[:, 1]) & (px[:, 1] == px[:, 2])).all()
    rec = doc.flourish_state(group).recipe
    for n in (1, 2, 3):
        counts = doc.apply_flourish(group, B.bake(dataclasses.replace(rec, seed=rec.seed + n)))
        assert counts.conflicts == 0, f"regenerate {n} flagged conflicts with nothing painted"


def test_flourish_cels_in_a_palette_constrained_document_stay_on_the_table():
    doc = Document.blank(24, 24)
    table = [(0, 0, 0, 255), (255, 255, 255, 255), RED]
    assert doc.set_palette(table)
    group = doc.insert_flourish(B.bake(_small()))
    allowed = {tuple(c[:3]) for c in table}
    for cel in _effect_cels(doc, group):
        px = cel.pixels[cel.pixels[..., 3] > 0]
        assert {tuple(int(v) for v in p[:3]) for p in px} <= allowed


# -- inker-13: switching the look -----------------------------------------------------------------


def test_regenerating_an_effect_in_the_other_look_leaves_no_tracks_of_the_old_one():
    doc = Document.blank(24, 24)
    group = doc.insert_flourish(B.bake(_small()))
    old = set(doc.flourish_state(group).tracks.values())
    assert len(old) > 1
    pixel = dataclasses.replace(doc.flourish_state(group).recipe, mode="pixel", colors=6)
    doc.apply_flourish(group, B.bake(pixel))
    state = doc.flourish_state(group)
    assert list(state.tracks) == [_doc_flourish.COMPOSITE]
    live = {track.uid for track in doc.anim.tracks}
    assert not (old & live), "the painterly tracks must be gone"
    assert not any(track in old for (track, _f) in doc.anim.cels)
    members = {uid for uid, parent in doc.group_of.items() if parent == group}
    assert members == set(state.tracks.values())
    # One step: Ctrl+Z puts the painterly look back whole.
    doc.undo()
    assert old <= {track.uid for track in doc.anim.tracks}
    doc.redo()
    assert not (old & {track.uid for track in doc.anim.tracks})


def test_switching_the_look_keeps_a_track_somebody_painted_on_and_flags_it():
    doc = Document.blank(24, 24)
    group = doc.insert_flourish(B.bake(_small()))
    state = doc.flourish_state(group)
    track_uid, frame = next(
        (t, f)
        for t in state.tracks.values()
        for f in doc.anim.frames
        if (t, f.uid) in doc.anim.cels
    )
    doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0] = (1, 2, 3, 255)
    pixel = dataclasses.replace(state.recipe, mode="pixel", colors=6)
    counts = doc.apply_flourish(group, B.bake(pixel))
    assert counts.conflicts >= 1
    assert track_uid in {track.uid for track in doc.anim.tracks}
    assert (track_uid, frame.uid) in doc.flourish_state(group).conflicts
    # "Replace painted cels" is the person's say-so to take it.
    doc.apply_flourish(group, B.bake(pixel), force=True)
    assert track_uid in {track.uid for track in doc.anim.tracks}, (
        "a track let go of is an ordinary layer now; force only governs tracks still in use"
    )


# -- inker-25: two effects' tags ------------------------------------------------------------------


def test_regenerating_one_effect_leaves_another_effects_same_named_tags_alone():
    doc = Document.blank(24, 24)
    a = doc.insert_flourish(B.bake(_small("explosion")))
    b = doc.insert_flourish(B.bake(_small("explosion")))
    names = [tag.name for tag in doc.anim.tags]
    assert len(names) == len(set(names)), "two effects must not share a tag name"
    spans_b = {t.name: (t.start, t.end) for t in doc.anim.tags if t.name.endswith("_2")}
    assert spans_b
    rec = doc.flourish_state(a).recipe
    phases = list(rec.phases)
    phases[0] = dataclasses.replace(phases[0], frames=phases[0].frames + 5)
    doc.apply_flourish(a, B.bake(dataclasses.replace(rec, phases=tuple(phases))))
    after_b = {t.name: (t.start, t.end) for t in doc.anim.tags if t.name.endswith("_2")}
    assert after_b == spans_b
    first = [t for t in doc.anim.tags if not t.name.endswith("_2")]
    assert (first[0].start, first[0].end) == (0, phases[0].frames - 1)
    assert doc.flourish_state(b).tag_suffix == "_2"


# -- inker-26 / inker-27 (document-10, flourish-05): the conflict memo --------------------------


def _doc_with_a_conflict():
    doc = Document.blank(24, 24)
    rec = _small()
    group = doc.insert_flourish(B.bake(rec))
    state = doc.flourish_state(group)
    owned = set(state.tracks.values())
    frame = doc.anim.frames[4]
    cel = next(c for (t, f), c in doc.anim.cels.items() if t in owned and f == frame.uid)
    cel.pixels[0, 0] = (1, 2, 3, 255)
    doc.apply_flourish(group, B.bake(dataclasses.replace(rec, seed=rec.seed + 1)))
    return doc, group, frame.uid


def _true_indices(doc, group):
    flagged = {f for _t, f in doc.flourish_state(group).conflicts}
    return sorted(i for i, frame in enumerate(doc.anim.frames) if frame.uid in flagged)


def test_flourish_conflict_indices_follow_a_frame_reorder():
    doc, group, _uid = _doc_with_a_conflict()
    assert doc.flourish_conflicts(group) == _true_indices(doc, group)
    assert doc.move_range(4, 4, len(doc.anim.frames) - 1)
    assert doc.flourish_conflicts(group) == _true_indices(doc, group)
    assert doc.reverse_range(0, len(doc.anim.frames) - 1)
    assert doc.flourish_conflicts(group) == _true_indices(doc, group)


def test_flourish_conflicts_follows_a_moved_frame():
    doc, group, uid = _doc_with_a_conflict()
    before = doc.flourish_conflicts(group)
    assert before
    assert doc.move_frame(before[0], len(doc.anim.frames) - 1)
    after = doc.flourish_conflicts(group)
    assert after == _true_indices(doc, group) != before
    assert doc.resolve_flourish(group, after)
    assert doc.flourish_conflicts(group) == []


# -- inker-27 (flourish-06): the frame rate -----------------------------------------------------


def test_changing_the_effect_fps_changes_the_durations_of_the_frames_it_covers():
    doc = Document.blank(24, 24)
    group = doc.insert_flourish(B.bake(_small(fps=18)))
    assert {f.duration_ms for f in doc.anim.frames} == {56}, "the pre-existing first frame too"
    rec = doc.flourish_state(group).recipe
    doc.apply_flourish(group, B.bake(dataclasses.replace(rec, fps=30)))
    assert {f.duration_ms for f in doc.anim.frames} == {33}
    doc.undo()
    assert {f.duration_ms for f in doc.anim.frames} == {56}


def test_a_regenerate_that_leaves_the_rate_alone_keeps_a_hand_set_duration():
    doc = Document.blank(24, 24)
    group = doc.insert_flourish(B.bake(_small(fps=18)))
    assert doc.set_frame_duration(3, 500)
    rec = doc.flourish_state(group).recipe
    doc.apply_flourish(group, B.bake(dataclasses.replace(rec, seed=rec.seed + 1)))
    assert doc.anim.frames[3].duration_ms == 500


# -- inker-10: redo of an indexed re-conversion --------------------------------------------------


def _indexed_rgb() -> Document:
    doc = Document.blank(3, 1)
    for x, colour in enumerate((RED, GREEN, BLUE)):
        doc.stack[0].pixels[0, x] = colour
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED, GREEN, BLUE], "nearest", transparent=0)
    return doc


def _row(doc: Document):
    return [tuple(int(v) for v in p) for p in doc.stack[0].pixels[0]]


def test_redo_of_a_synchronous_reconvert_of_an_indexed_document_reproduces_the_first_result():
    doc = _indexed_rgb()
    assert doc.set_palette([HOLE, GREEN, BLUE, RED])
    first = _row(doc)
    first_indices = doc.stack[0].indices.copy()
    assert first == [RED, GREEN, BLUE], "nearest keeps the colours"
    doc.undo()
    doc.redo()
    assert _row(doc) == first
    assert np.array_equal(doc.stack[0].indices, first_indices)
    for _ in range(3):  # and it stays stable however often it is walked
        doc.undo()
        doc.redo()
    assert _row(doc) == first


# -- inker-23: remap_palette ----------------------------------------------------------------------


def test_remap_palette_swaps_two_slots_colours_and_leaves_the_index_plane_alone():
    doc = _indexed_rgb()
    before = doc.stack[0].indices.copy()
    assert doc.remap_palette([HOLE, GREEN, RED, BLUE])
    assert np.array_equal(doc.stack[0].indices, before)
    assert _row(doc) == [GREEN, RED, BLUE]
    doc.undo()
    assert _row(doc) == [RED, GREEN, BLUE]
    doc.redo()
    assert _row(doc) == [GREEN, RED, BLUE]


def test_remap_palette_refuses_a_table_that_loses_a_slot_the_drawing_uses():
    doc = _indexed_rgb()
    with pytest.raises(ValueError):
        doc.remap_palette([HOLE, GREEN])


# -- inker-08: flatten and redo -------------------------------------------------------------------


def test_redoing_back_to_the_state_before_a_flatten_restores_the_same_picture_and_never_raises():
    doc = Document.blank(4, 4)
    art = np.zeros((4, 4, 4), np.uint8)
    art[0, 0] = RED
    doc.put_clipboard(art)
    assert doc.paste_as_layer()
    doc.flip("horizontal")
    before = doc.flatten().copy()
    doc.flatten_layers()
    for _ in range(3):  # the flatten, the flip, the paste
        assert doc.undo()
    for _ in range(2):  # the paste, the flip
        assert doc.redo()
    assert np.array_equal(doc.flatten(), before)
    # And all the way back round, flatten included.
    assert doc.redo()
    assert doc.undo()
    assert np.array_equal(doc.flatten(), before)


def test_redoing_past_a_flatten_after_a_canvas_resize_never_raises():
    doc = Document.blank(6, 6)
    doc.add_layer("b")
    doc.crop((0, 0, 3, 3))
    doc.flatten_layers()
    while doc.history.can_undo:
        doc.undo()
    assert doc.size == (6, 6)
    while doc.history.can_redo:
        assert doc.redo()
    assert doc.size == (3, 3) and len(doc.stack) == 1


def test_an_animated_flatten_walks_undo_and_redo_without_raising():
    doc = Document.blank(6, 6)
    doc.add_layer("b")
    doc.stack[0].pixels[1, 1] = RED
    doc.stack[1].pixels[2, 2] = GREEN
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(copy=True)
    doc.history.clear()
    doc.duplicate_layer(0)
    doc.crop((0, 0, 3, 3))
    doc.flatten_layers()
    while doc.history.can_undo:
        doc.undo()
    assert doc.size == (6, 6)
    while doc.history.can_redo:
        assert doc.redo()
    assert doc.size == (3, 3) and len(doc.anim.tracks) == 1


class _Boom(Edit):
    def undo(self, doc):  # pragma: no cover - never undone here
        pass

    def redo(self, doc):
        raise ValueError("boom")


def test_a_redo_that_raises_leaves_the_step_on_the_redo_stack():
    stack = UndoStack()
    edit = _Boom()
    stack.push(edit)
    stack._done.pop()
    stack._undone.append(edit)
    with pytest.raises(ValueError):
        stack.redo(object())
    assert stack.can_redo, "the step must not be lost with the raise"
    assert stack.redo_top is edit


# -- inker-09: merge down over an empty lower slot ------------------------------------------------


def test_undoing_a_merge_down_over_an_empty_lower_slot_leaves_no_cel_opacity_behind():
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.add_layer("top")
    doc.stack[1].pixels[0, 0] = GREEN
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(copy=True)
    doc.clear_cel(0, 1)
    doc.history.clear()
    assert doc.merge_down(1)
    doc.undo()
    assert doc.anim.cel_opacity == {}
    # ... so what is drawn on that slot next is still visible after a rebuild.
    doc.set_current_frame(1)
    doc.set_active_layer(0)
    assert doc.begin_stroke((1, 1), (0, 0, 255, 255), size=1, hardness=1.0, nib="square")
    doc.stroke_to((2, 2))
    doc.end_stroke()
    doc.set_current_frame(0)
    doc.set_current_frame(1)
    assert int(doc.flatten(matte=False)[1, 1, 3]) == 255


# -- inker-24: the convert preview's stamps -------------------------------------------------------


def test_cancelling_a_convert_preview_restamps_frames_that_share_the_previewed_cel():
    doc = Document.blank(2, 1)
    doc.stack[0].pixels[0, 0] = (200, 40, 40, 255)
    doc.stack[0].pixels[0, 1] = (40, 40, 200, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    other = doc.anim.frames[1].uid
    original = doc.frame_flat(other)[0, 0].tolist()
    assert doc.begin_convert()
    assert doc.preview_convert([(0, 0, 0, 255), (255, 255, 255, 255)], "nearest")
    doc._frame_cache.clear()
    doc._frame_order.clear()
    assert doc.frame_flat(other)[0, 0].tolist() != original, "the preview reaches the neighbour"
    assert doc.cancel_convert()
    assert doc.frame_flat(other)[0, 0].tolist() == original


# -- inker-38: the timeline's link query ----------------------------------------------------------


class _CountingCels(dict):
    """A ``cels`` dict that counts how many times it is walked."""

    walks = 0

    def values(self):
        type(self).walks += 1
        return super().values()

    def items(self):
        type(self).walks += 1
        return super().items()


def test_is_linked_cost_is_linear_in_the_number_of_cels_for_a_whole_grid_pass():
    doc = Document.blank(4, 4)
    for _ in range(9):
        doc.add_layer("x")
    doc.ensure_animation()
    for _ in range(39):
        doc.add_frame(link=True)
    anim = doc.anim
    counting = _CountingCels(anim.cels)
    anim.cels = counting
    _CountingCels.walks = 0
    linked = anim.linked_slots()
    assert _CountingCels.walks <= 2, "the whole grid is answered in one pass"
    expected = {
        (t.uid, f.uid) for t in anim.tracks for f in anim.frames if anim.is_linked(t.uid, f.uid)
    }
    assert set(linked) == expected and linked
    # A grid of this size asked cell by cell would scan the cels once per cell.
    start = time.perf_counter()
    anim.linked_slots()
    assert time.perf_counter() - start < 0.5


def test_the_timeline_asks_for_every_link_once_per_draw_not_once_per_cell():
    from realmspinner.studio.modes.inker.ui.panes import timeline

    grid = inspect.getsource(timeline._grid)
    assert "linked_slots" in grid
    cell = ast.parse(inspect.getsource(timeline._cell).lstrip())
    reads = [
        node
        for node in ast.walk(cell)
        if isinstance(node, ast.Constant) and node.value == "linked"
    ]
    assert reads, "_cell must read the per-draw link set out of geom"
