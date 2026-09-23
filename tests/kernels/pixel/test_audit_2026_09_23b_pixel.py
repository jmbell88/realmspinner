"""Regression tests for the 2026-09-23 (second run) audit's inker fixer
batch, for the findings whose owned files live under ``kernels/pixel``.

inker-01 (``_doc_layers.move_into_group`` letting a ``ValueError`` from a
background-layer reorder escape with the membership already changed),
inker-03 (``ora._read_animation``'s uncapped ``cels`` list) and inker-04
(``filters.remove_orphans`` with no cost ceiling) live here. inker-02 and
inker-05 (``studio/modes/inker``) live in
``tests/modes/inker/test_audit_2026_09_23b_inker.py``.
"""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np

from realmspinner.kernels.pixel import filters, ora
from realmspinner.kernels.pixel import groups as gp
from realmspinner.kernels.pixel.document import Document

# -- inker-01: move_into_group and a refused reorder -----------------------------------------


def _doc(layers: int = 4) -> Document:
    """``tests/modes/inker/test_groups.py``'s fixture shape, not imported
    from there because that module is not one of this batch's owned files."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "L0"
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
    doc.invalidate_all()
    return doc


def _ok(doc: Document) -> None:
    gp.check(doc.groups, doc.group_of, doc.member_uids())


def test_move_into_group_refuses_a_reorder_that_would_strand_the_background_layer():
    """The 2026-09-23 (second run) audit, finding inker-01:
    ``LayerStack.move`` refuses (``ValueError``) a reorder that would leave a
    background layer off the bottom row (inker-02, 90af9bba) and
    ``move_layer`` already catches that and refuses in kind. This door --
    reached by dragging a layer onto a group folder -- set the membership
    *before* attempting the move, so the same ``ValueError`` escaped here
    uncaught: the background layer's ``group_of`` went from ``None`` to the
    group with no undo step pushed for it, altering the document with
    nothing to undo.
    """
    doc = _doc(4)
    assert doc.to_background()  # L0 becomes the background layer
    node = doc.group_layers([1, 2], name="Ink")
    assert node is not None
    _ok(doc)

    background_uid = doc.stack[0].uid
    assert doc.group_of.get(background_uid) is None
    history_depth = len(doc.history)

    # Dragging the background layer (index 0) onto the group would have to
    # move it off the bottom row to land adjacent to the group's span --
    # exactly the reorder ``LayerStack.move`` refuses.
    moved = doc.move_into_group(0, node.uid)

    assert moved is False
    assert doc.group_of.get(background_uid) is None
    assert len(doc.history) == history_depth
    assert doc.stack[0].background is True
    _ok(doc)


# -- inker-03: an unbounded ``cels`` list in animation.json ------------------------------------


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def _ora_zip_naming_one_plane(cel_count: int) -> zipfile.ZipFile:
    """An ``animation.json`` whose ``cels`` list is ``cel_count`` entries
    long, every one of them a *valid* cel naming the same plane -- the shape
    the finding describes ("2,000,000 entries naming one plane"). Every
    entry decodes and links cleanly, so without the ceiling this reads all
    the way through instead of failing for some unrelated reason -- the
    ceiling has to be what stops it.
    """
    payload = {
        "version": ora.ANIMATION_VERSION,
        "tracks": [{"name": "L0"}],
        "frames": [{"duration_ms": 100}],
        "cels": [{"track": 0, "frame": 0, "data": "layer0.png"} for _ in range(cel_count)],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(ora.ANIMATION_MEMBER, json.dumps(payload))
        zf.writestr("layer0.png", _png_bytes())
    buf.seek(0)
    return zipfile.ZipFile(buf, "r")


def test_an_animation_with_millions_of_cels_naming_one_plane_is_refused_before_the_loop():
    """The 2026-09-23 (second run) audit, finding inker-03: every sibling
    list in ``animation.json`` (tracks, frames, a frame's palette) is capped,
    but ``cels`` was not -- 2,000,000 entries naming one plane cost 2.9s. A
    count past the ceiling must be refused (``None``, degrading to the flat
    read) rather than read through to a real grid.
    """
    zf = _ora_zip_naming_one_plane(ora.MAX_ORA_METADATA_ENTRIES + 1)
    assert ora._read_animation(zf, (8, 8)) is None

    # A count at the ceiling is unaffected -- this reads through to a real
    # grid, not the ``None`` degrade the ceiling forces past it.
    zf_ok = _ora_zip_naming_one_plane(ora.MAX_ORA_METADATA_ENTRIES)
    assert ora._read_animation(zf_ok, (8, 8)) is not None


# -- inker-04: remove_orphans with no cost ceiling on a busy layer -----------------------------


def test_remove_orphans_on_a_busy_layer_does_not_stall_the_frame_thread():
    """The 2026-09-23 (second run) audit, finding inker-04: no cost ceiling,
    unlike every sibling matte filter, and it runs on the frame thread
    through ``Document.preview_filter``: 0.5s at 512^2 and 2.06s at 1024^2 on
    a busy (noisy) layer, where almost every opaque pixel is friendless.
    """
    rng = np.random.default_rng(0)
    size = 1024
    field = np.zeros((size, size, 4), dtype=np.uint8)
    # A wide colour space so an 8-neighbour almost never shares a pixel's
    # exact colour -- the worst case for the per-lonely-pixel Python loop.
    field[..., :3] = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    field[..., 3] = 255

    start = time.perf_counter()
    filters.remove_orphans(field, orphans=1.0)
    elapsed = time.perf_counter() - start

    # Comfortably under the unfixed 2.06s at this size, and short enough not
    # to read as the frame thread hanging under a live preview.
    assert elapsed < 1.0
