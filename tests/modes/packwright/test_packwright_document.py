"""``PackDoc`` itself: adding, removing, renaming and replacing a source.

No such module existed before the 2026-09-07 audit's packwright-01 -- the
document's mechanics were exercised only indirectly, through ``test_pack_meta``
(metadata plumbing) and ``test_wpack`` (the file format). This file is where a
door on ``PackDoc`` gets a test of its own, starting with the one the audit
found had none: the pixel ceiling ``.wpack`` enforces on the way back in but
nothing enforced on the way in.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.modes.packwright.engine.document import PackDoc
from warlock.studio.modes.packwright.engine.sources import Sprite


def test_a_document_with_a_source_past_max_source_pixels_is_refused_at_add_not_only_on_reopen(
    monkeypatch,
):
    """packwright-01: ``MAX_SOURCE_PIXELS`` (16,000,000) was enforced only in
    ``read_wpack``'s ``_pixels_from`` -- ``PackDoc.add_source`` refused a
    duplicate key and a full pack but had no pixel ceiling, and neither did
    ``replace_source``. A 4096x4096 sprite (16,777,216 pixels) was accepted
    here, written to a ``.wpack`` by ``wpack_bytes``, and refused only when
    Warlock tried to reopen the file it had just written -- a document
    Warlock itself authored that Warlock itself could never read back, which
    breaks the reopenable-exports contract (INVARIANTS:215) with no help from
    a hand-edited file at all.
    """
    from warlock.studio.modes.packwright.engine import wpack

    # The audit's own numbers: one hair past the real ceiling, refused right
    # here rather than written first.
    doc = PackDoc()
    huge = Sprite(key="a", name="a", pixels=np.zeros((4096, 4096, 4), dtype=np.uint8))
    assert huge.width * huge.height > wpack.MAX_SOURCE_PIXELS
    with pytest.raises(ValueError, match="the atlas format's limit"):
        doc.add_source(huge)
    assert doc.sources == [], "refused at the door, not left half-added"

    # The same ceiling refuses a *replacement*, not only a fresh add --
    # exercised at a monkeypatched ceiling so this half does not also
    # allocate 64 MB to make its point.
    monkeypatch.setattr(wpack, "MAX_SOURCE_PIXELS", 100)
    doc.add_source(Sprite(key="b", name="b", pixels=np.zeros((4, 4, 4), dtype=np.uint8)))
    uid = doc.sources[0].uid
    with pytest.raises(ValueError, match="the atlas format's limit is 100 pixels"):
        doc.replace_source(
            uid, Sprite(key="b", name="b", pixels=np.zeros((11, 11, 4), dtype=np.uint8))
        )
    assert doc.sources[0].sprite.width == 4, "the refused replacement did not land"


def test_a_document_of_many_near_ceiling_sprites_is_refused_before_the_aggregate_pixel_budget_is_allocated(  # noqa: E501
    monkeypatch,
):
    """packwright-01 (2026-09-13 audit): ``MAX_SOURCE_PIXELS`` bounds one
    sprite and ``MAX_DECOMPRESSED_BYTES`` bounds the archive's stored PNG
    bytes -- neither bounds the *sum* of every sprite a document holds, so a
    document of many sprites each individually under the per-sprite ceiling
    had no ceiling on their total at all. Exercised at a monkeypatched
    document budget so this proves refusal without allocating anywhere near
    the real one (8192 squared)."""
    from warlock.studio.modes.packwright.engine import wpack

    monkeypatch.setattr(wpack, "MAX_DOCUMENT_PIXELS", 100)
    doc = PackDoc()
    # Each sprite is 40 pixels, comfortably under any per-sprite ceiling; two
    # of them (80) still fit the patched document budget of 100, a third
    # (120) does not.
    doc.add_source(Sprite(key="a", name="a", pixels=np.zeros((5, 8, 4), dtype=np.uint8)))
    doc.add_source(Sprite(key="b", name="b", pixels=np.zeros((5, 8, 4), dtype=np.uint8)))
    assert doc.total_pixels() == 80
    with pytest.raises(ValueError, match="document-wide limit is 100 pixels"):
        doc.add_source(Sprite(key="c", name="c", pixels=np.zeros((5, 8, 4), dtype=np.uint8)))
    assert {s.key for s in doc.sources} == {"a", "b"}, "refused at the door, not left half-added"

    # A replacement is checked the same way, with its own source's pixels
    # backed out of the running total first -- replacing "a" with a
    # same-sized sprite must not be refused for a budget the document already
    # holds under its old reading of "a".
    uid = doc.sources[0].uid
    same_size = np.zeros((5, 8, 4), dtype=np.uint8) + 1
    doc.replace_source(uid, Sprite(key="a", name="a", pixels=same_size))
    with pytest.raises(ValueError, match="document-wide limit is 100 pixels"):
        doc.replace_source(
            uid, Sprite(key="a", name="a", pixels=np.zeros((9, 8, 4), dtype=np.uint8))
        )
