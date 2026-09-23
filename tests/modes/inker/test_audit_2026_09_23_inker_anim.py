"""Follow-up to the 2026-09-23 audit's inker-02 fix: the animated path.

``tests/modes/inker/test_audit_2026_09_23_inker.py`` pins the still-document
refusal (``LayerStack.move`` and ``Document.move_layer`` on a still stack),
but ``Document.move_layer`` calls ``Document._move_track`` directly for an
animated document (``_move_row_edit``, ``kernels/pixel/_doc_layers.py``)
rather than routing through ``self.stack.move`` -- so the still-document
refusal never reached a Track Grid reorder and a drag there could still
strand a background track mid-stack. This file is the same invariant, pinned
on the animated path.
"""

from __future__ import annotations

from realmspinner.kernels.pixel.document import Document


def _animated_bg_doc() -> Document:
    doc = Document.blank(4, 4)
    doc.add_layer()
    assert doc.to_background() is True
    doc.ensure_animation()
    assert doc.has_background is True
    assert [t.background for t in doc.anim.tracks] == [True, False]
    return doc


def test_moving_a_background_track_off_the_bottom_is_refused():
    doc = _animated_bg_doc()

    moved = doc.move_layer(0, 1)

    assert moved is False, "a drag that would strand the background mid-stack must refuse"
    assert [t.background for t in doc.anim.tracks] == [True, False]
    assert doc.has_background is True


def test_moving_another_track_to_the_bottom_of_a_background_grid_is_refused():
    """The other direction: dragging an ordinary track under the background
    one displaces the background track to row 1 just as surely as dragging
    the background track itself does."""
    doc = _animated_bg_doc()

    moved = doc.move_layer(1, 0)

    assert moved is False
    assert [t.background for t in doc.anim.tracks] == [True, False]
    assert doc.has_background is True


def test_move_track_itself_refuses_the_same_reorder_and_leaves_tracks_untouched():
    """Pin the refusal at ``Document._move_track`` directly, the same way
    ``test_layer_stack_move_itself_refuses_the_same_reorder`` pins it at
    ``LayerStack.move`` for the still path -- and prove a refused move is
    atomic: ``anim.tracks`` must come back exactly as it went in."""
    import pytest

    doc = _animated_bg_doc()
    tracks_before = list(doc.anim.tracks)
    bg_uid = doc.anim.tracks[0].uid

    with pytest.raises(ValueError):
        doc._move_track(bg_uid, 1)

    assert doc.anim.tracks == tracks_before
    assert doc.anim.tracks[0].uid == bg_uid
