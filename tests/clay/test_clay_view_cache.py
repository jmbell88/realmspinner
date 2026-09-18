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
this module cannot live beside, since ``tests/clay/`` is this fix's allowed
test directory and that file is not in it) already builds the GL-backed
``view``/``_doc`` this needs, the same way ``test_undo_gesture_doors.py``
imports ``test_sirens_panes_smoke``'s ``frames`` fixture rather than a second
copy of it.
"""

from __future__ import annotations

import gc
import weakref

from test_clay_view import RECT, _doc
from test_clay_view import view as view  # noqa: F401, PLC0414


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
