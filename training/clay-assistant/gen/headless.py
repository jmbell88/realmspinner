"""``HeadlessCtx``: the ctx double this dataset tooling drives ``agent_clay``
through, plus a non-pytest way to open a GL context for the gallery.

``HeadlessCtx`` is copied -- not imported -- from ``tests/test_agent_clay.py``'s
``_Ctx`` (see that file's own module docstring, "a ctx double, no imgui, no
GL, no pygame"): importing from ``tests/`` would make a data-generation tool
depend on the test tree, which is backwards from every other dependency this
repo draws (``service`` and ``studio`` never import ``tests``, and neither
should ``training/``). A change to ``_Ctx`` there is meant to be found by
grep on this docstring, not by an import that would make ``training/`` a
second, silent consumer of the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class HeadlessCtx:
    """Exactly what ``agent_clay`` and ``clay_mode`` read off ``ctx`` -- see
    ``tests/test_agent_clay.py::_Ctx`` for the original and why each
    attribute exists. No ``viewer``: ``clay_render`` is the one tool this
    omission is meant to exercise, and no training row ever calls it (see
    ``gen/verify.py``'s own docstring).
    """

    def __init__(self, svc: Any = None) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = svc
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        # ``clay_mode.adopt`` calls ``remember_path(ctx, path)`` on every new
        # document, and ``path`` is always ``None`` here (a generated
        # record's document has no file) -- so ``recents.remember`` no-ops
        # before ever reading this, but the attribute access itself
        # (``ctx.settings``) still has to succeed for that no-op to be
        # reached at all.
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def open_gl() -> Any | None:
    """A standalone GL 3.3 context, or ``None`` if there is no driver for
    one -- the non-pytest twin of ``tests/conftest.py``'s session-scoped
    ``gl`` fixture (see that fixture's own docstring for why context creation
    is skipped rather than failed with no GPU). Used by ``build.py --gallery``
    and by ``render.py``; nothing here needs a window or pygame.
    """
    try:
        import moderngl
    except Exception:
        return None
    try:
        return moderngl.create_context(standalone=True, require=330)
    except Exception:
        return None
