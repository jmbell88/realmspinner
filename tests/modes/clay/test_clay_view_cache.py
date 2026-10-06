"""Clay's viewport render cache: the document has to be pinned, not just its id.

The 2026-09-07 audit's clay-09. ``ClayView.draw``'s frame-skip key
(``studio/modes/clay/ui/view.py``) carried ``id(doc)`` with nothing holding a reference to
*that* document -- unlike every other identity-keyed cache in the class
(``_screens``, ``_world_cache``), which already pin the object an id in their
key names, for the reason both of their own comments state: an id is only
sound while its object is alive. With nothing pinning it here, a closed tab's
``ClayDoc`` could be collected and a new document minted at the same address --
and with ``rev`` starting at 0 for every fresh document, a cache entry from the
old one could read as still current and hand back its stale texture.

Reused fixtures rather than a second copy: ``test_clay_view.py`` (the sibling
this module cannot live beside, since ``tests/modes/clay/`` is this fix's allowed
test directory and that file is not in it) already builds the GL-backed
``view``/``_doc`` this needs, the same way ``test_undo_gesture_doors.py``
imports ``test_sirens_panes_smoke``'s ``frames`` fixture rather than a second
copy of it.
"""

from __future__ import annotations

import gc
import weakref

from .test_clay_view import RECT, _doc
from .test_clay_view import view as view  # noqa: F401, PLC0414


def test_a_draw_pins_the_document_it_rendered(view) -> None:
    """The direct claim: after a draw, ``ClayView`` itself holds a strong
    reference to the document just rendered, not only its id."""
    doc = _doc(count=1)
    view.draw(doc, RECT, 0.0)
    assert view._last_doc is doc


def test_the_pinned_document_survives_every_other_reference_being_dropped(view) -> None:
    """The mechanism the finding is actually about: without the pin, nothing
    stops the allocator from reusing a closed document's address for a new
    one, and a stale cache entry then reads as current. Proven directly
    against the real ``draw`` path rather than by trying to force an id
    collision (inherently non-deterministic): once every other reference to
    ``doc`` is gone -- the tab closed, in the app -- a weak reference to it
    must still resolve, because ``view`` is the one reference left.
    """
    doc = _doc(count=1)
    view.draw(doc, RECT, 0.0)
    ref = weakref.ref(doc)

    del doc
    gc.collect()

    assert ref() is not None, "the pin must keep the document alive"
    view._last_doc = None
    gc.collect()
    assert ref() is None, "and release it once the view stops pointing at it"


def test_world_cache_for_one_uid_is_not_shared_between_a_document_and_its_preview_scratch(
    view,
) -> None:
    """The 2026-09-20 audit's clay-20: ``ClayView._world``'s own memo
    (``_world_cache``) keyed on ``obj.uid`` alone -- unlike ``_centre_memo``/
    ``_bounds_memo``, this module's header names as the sibling pattern,
    which key on ``(id(doc), uid)``. ``_ghost_draws`` calls ``_world``
    against two documents that deliberately share a uid namespace: the live
    document and its Familiar preview scratch clone. One dict slot per uid
    meant the second document queried in a frame evicted the first's entry,
    every frame both a document and its own preview are drawn, defeating the
    very pin that is supposed to keep an object's transform arrays alive for
    the memo to trust.
    """
    doc = _doc(count=1)
    scratch = _doc(count=1)
    # A Familiar preview clone starts from a full copy of ``doc``, uid and
    # all -- forced directly here rather than going through ``clay.scratch``,
    # since the uid collision between two distinct documents is the only
    # part of that clone this test needs.
    scratch.objects[0].uid = doc.objects[0].uid

    view._world(doc, doc.objects[0])
    view._world(scratch, scratch.objects[0])

    assert len(view._world_cache) == 2, (
        "one document's world-matrix entry evicted the other's -- same uid, "
        "different document, one shared dict slot"
    )
