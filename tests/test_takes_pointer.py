"""``shell.frame._takes_pointer``: the one hover/grab rule shared by all three
viewports.

The 2026-09-11 audit's clay-02 fix renamed ``ClayView``'s own "any grab is
live" property from ``dragging`` (which shadowed ``DragOps.dragging``, a
narrower "is a transform running" meaning) to ``grabbing``. ``_takes_pointer``
was reading ``target.dragging`` unconditionally, so a Clay viewport orbiting,
panning or marqueeing with the cursor off its rect would drop pointer capture
the instant the narrow ``dragging`` (correctly) answered ``False`` -- the
capture rule needs the *broad* answer, which now lives under a different
name for Clay and under ``dragging`` for the asset viewer and Poser's, which
never had the shadowing property to begin with.

The P4 restructure (``dev/RESTRUCTURE.md``) moved this function out of
``studio/main.py`` into ``studio/shell/frame.py``, where every workspace
that reads it now reaches for it by name (``from .shell.frame import
_takes_pointer``) instead of the bare module-global name ``main.py`` used to
resolve for free.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio.shell import frame


def test_a_clayview_shaped_target_keeps_capture_by_its_broad_grabbing_property():
    """A ``ClayView`` stub: ``_grab = "orbit"`` (a camera gesture, not a
    transform) so its narrow ``dragging`` -- if it had one -- would say
    False, but its ``grabbing`` says True. Capture must follow ``grabbing``."""
    target = SimpleNamespace(grabbing=True)
    assert frame._takes_pointer(target, False) is True

    target = SimpleNamespace(grabbing=False)
    assert frame._takes_pointer(target, False) is False


def test_a_viewer_with_only_dragging_still_works():
    """The asset viewer and Poser's never had the shadowing property, so
    ``dragging`` alone must still be read for them."""
    assert frame._takes_pointer(SimpleNamespace(dragging=True), False) is True
    assert frame._takes_pointer(SimpleNamespace(dragging=False), False) is False


def test_hovered_or_none_still_behave():
    assert frame._takes_pointer(None, True) is True
    assert frame._takes_pointer(None, False) is False
