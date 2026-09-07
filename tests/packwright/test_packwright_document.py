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

from warlock.studio.packwright.document import PackDoc
from warlock.studio.packwright.sources import Sprite


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
    from warlock.studio.packwright import wpack

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
