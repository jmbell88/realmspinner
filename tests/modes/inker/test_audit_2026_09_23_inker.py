"""Regressions for the 2026-09-23 audit's Inker codec findings.

Four unrelated doors, one file because the audit grouped them: an .ora reader
with an uncapped metadata map (inker-01), a reorder that could leave a
background layer off the bottom of the stack (inker-02), a palette door with
no ceiling unlike its siblings (inker-05), and a GIF export that leaked
already-built PIL images on a mid-build failure (inker-07).
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.pixel import gifout
from realmspinner.kernels.pixel import ora as inker_ora
from realmspinner.kernels.pixel.animation import Animation, Frame, Track
from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.groups import GroupNode
from realmspinner.kernels.pixel.layers import Layer, LayerStack
from realmspinner.kernels.pixel.undo import UNDO_BYTES, UndoStack

HOLE = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


# --- inker-01: ora.py's _read_flourish "assets" map had no ceiling ----------


def _flourish_doc():
    """A minimal one-track, one-frame animated document with one layer group,
    the same shape the 2026-09-23 audit's ``inker-codecs-01.py`` probe built,
    so ``_read_flourish`` has something real to attach an entry to."""
    layer = Layer.empty(8, 8, "Ink")
    track = Track(name="Ink")
    frame = Frame()
    anim = Animation(tracks=[track], frames=[frame], cels={(track.uid, frame.uid): layer})
    doc = Document(
        stack=LayerStack(anim.layers_for(frame, (8, 8)), 0),
        history=UndoStack(UNDO_BYTES),
        anim=anim,
    )
    node = GroupNode(name="Group")
    doc.groups[node.uid] = node
    doc.group_of[track.uid] = node.uid
    return doc, node


def test_read_flourish_refuses_an_assets_map_past_the_ceiling():
    """The 2026-09-23 audit, finding inker-01: "assets" had no
    ``MAX_ORA_METADATA_ENTRIES`` ceiling, unlike its "tracks"/"digests"/
    "conflicts" siblings capped in the same function for the same reason
    (the 2026-09-19 audit, finding inker-05) -- one real PNG named under many
    ids costs one decode and one RGBA copy per id on open."""
    doc, node = _flourish_doc()
    n = inker_ora.MAX_ORA_METADATA_ENTRIES
    payload = {
        "flourish": [
            {
                "group": 0,
                "recipe": {"kind": "unknown-recipe-kind-so-nothing-decodes"},
                "tracks": {},
                "digests": [],
                "conflicts": [],
                "offset": [0, 0],
                "assets": {str(i): "data/flourish0_x.png" for i in range(n + 1)},
            }
        ],
    }
    inker_ora._read_flourish(doc, payload, [node])
    # The malformed entry is dropped whole -- the same tolerant behaviour
    # every sibling ceiling in this function falls back to -- rather than the
    # oversized "assets" map being accepted and materialised.
    assert node.uid not in doc.flourish


# --- inker-02: a reorder could leave a background layer off the bottom -----


def test_moving_a_background_layer_off_the_bottom_is_refused():
    doc = Document.blank(4, 4)
    doc.add_layer()
    assert len(doc.stack) == 2
    assert doc.to_background() is True
    assert doc.has_background is True

    moved = doc.move_layer(0, 1)

    assert moved is False, "a drag that would strand the background mid-stack must refuse"
    assert doc.stack[0].background is True
    assert doc.has_background is True
    assert [layer.background for layer in doc.stack] == [True, False]


def test_moving_another_layer_to_the_bottom_of_a_background_document_is_refused():
    """The same invariant, the other direction: dragging an ordinary layer
    under the background one displaces the background layer to row 1 just as
    surely as dragging the background layer itself does."""
    doc = Document.blank(4, 4)
    doc.add_layer()
    doc.to_background()

    moved = doc.move_layer(1, 0)

    assert moved is False
    assert doc.stack[0].background is True


def test_layer_stack_move_itself_refuses_the_same_reorder():
    """LayerStack's own docstring (layers.py) claims it is "the one place
    that rule is enforced" -- pin the claim at the class the docstring is on,
    not only at the Document door above it."""
    bg = Layer.empty(4, 4, "Floor")
    bg.background = True
    top = Layer.empty(4, 4, "Top")
    stack = LayerStack([bg, top], active=0)

    with pytest.raises(ValueError):
        stack.move(0, 1)
    assert stack.layers[0] is bg, "a refused move must not mutate the stack"


# --- inker-05: set_frame_palette had no MAX_COLOURS ceiling -----------------


def _animated(frames: int = 2) -> Document:
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED], "nearest", transparent=0)
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def test_set_frame_palette_refuses_more_than_max_colours():
    from realmspinner.kernels.pixel import index_plane as ixp

    doc = _animated()
    too_many = [(i % 256, 0, 0, 255) for i in range(ixp.MAX_COLOURS + 1)]
    with pytest.raises(ValueError):
        doc.set_frame_palette(too_many, 1)
    # Refused at the door, so the frame is left with no override at all.
    assert doc.anim.frame_palettes == {}


# --- inker-07: write_gif leaked already-built PIL images on a refusal ------


def _plane(size, colour):
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[:, :] = colour
    return out


def test_write_gif_closes_already_built_frames_when_a_later_one_fails_to_convert(
    tmp_path, monkeypatch
):
    dest = tmp_path / "clip.gif"
    frame = _plane((4, 4), (10, 20, 30, 255))
    real_quantise = gifout.quantise
    closed = []
    calls = {"n": 0}

    def flaky_quantise(plane):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom -- simulating a mid-build failure")
        image = real_quantise(plane)
        original_close = image.close

        def tracking_close():
            closed.append(image)
            original_close()

        image.close = tracking_close
        return image

    monkeypatch.setattr(gifout, "quantise", flaky_quantise)

    with pytest.raises(ValueError, match="boom"):
        gifout.write_gif(dest, [frame, frame, frame], [100, 100, 100])

    assert len(closed) == 1, (
        "the one PIL image already built before the second frame's conversion "
        "failed must still be closed -- the old code's try/finally wrapped "
        "only .save(), so a failure while building the image list closed nothing"
    )
