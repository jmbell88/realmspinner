"""Mason's controller -- empty on purpose.

Mason is the fourteenth workspace mode, registered in ``modes.py``, but this
first stage gives it nothing to control yet: no engine package, no document
format, no panes. The mode opens on a bare workspace and does nothing. That is
deliberate rather than unfinished -- a document format, an undo stack and a
journal provider are all built against a concrete node tree, and writing any
of them against nothing would be guessing at a shape a later stage still gets
to choose.

:func:`handle_key` exists only so ``main.py``'s per-mode keyboard dispatch has
an arm for Mason the way every other workspace mode does, and so that arm can
``return`` unconditionally the way ``packwright_mode`` and ``muse_mode`` do --
without it, a key pressed in Mason would fall through to whichever other
mode's bindings the shared block still holds, and inherit keys that mean
something else entirely (Packwright's own ``packwright-02`` finding was a
version of exactly this: one mode's ``Delete`` reaching another mode's data).
Binding nothing here is the correct amount of behaviour for a mode with no
document to act on -- there is nothing yet for Ctrl+S to save, nothing for
Delete to remove, nothing for Ctrl+Z to undo.

No ``JOURNAL`` provider is registered here, and that is also deliberate:
crash recovery is a mechanism for reopening a document that existed a moment
ago, and Mason has no document to recover until a later stage gives it a
save format.
"""

from __future__ import annotations

from typing import Any


def handle_key(ctx: Any, event: Any) -> bool:
    """Mason's keyboard. Binds nothing, consumes nothing.

    Always ``False``: there is no document and no selection yet for any key to
    act on. The caller (``main.py``'s ``_shortcut``) returns unconditionally
    once it has dispatched to a workspace mode's ``handle_key``, whether or
    not that call consumed the key, so Mason returning ``False`` still stops
    the key from reaching the 2D/3D shared bindings meant for other modes.
    """
    return False
