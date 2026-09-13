"""The Familiar programme: an agent previews an edit on a scratch clone of
Clay's document before it ever touches the one the user is looking at.

See :mod:`~warlock.studio.clay.scratch` for the clone/diff/transplant
mechanics this package drives, and :mod:`.scratch_ctx` for the sandboxed
``ctx`` a scratch run executes an agent tool call against.
"""

from __future__ import annotations

from .apply import apply, discard
from .scratch_ctx import PREVIEW_EXCLUDED, ScratchCtx, build, run_scratch

__all__ = [
    "PREVIEW_EXCLUDED",
    "ScratchCtx",
    "apply",
    "build",
    "discard",
    "run_scratch",
]
