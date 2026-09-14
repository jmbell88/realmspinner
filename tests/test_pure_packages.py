"""``_pure_packages``: the helper every sibling-ban pin in ``studio/`` derives
its "which packages are headless" answer from.

Measured on 2026-09-14 while wiring T3's new pure modules into
``studio/familiar/``: the helper only ever looked at ``node.level == 0``
(an absolute import), so a relative import -- ``from .. import clay_mode``,
``from .. import agent_clay`` -- contributed nothing to a file's roots at
all, and a package that reached a window purely through relative imports
came back "pure" from :func:`_pure_packages.pure_packages`.
"""

from __future__ import annotations

from pathlib import Path

import _pure_packages as pp


def test_a_relative_import_of_a_gl_module_is_not_counted_pure(tmp_path: Path):
    """A tmp package whose only path to a window is two relative hops away
    (mirroring ``familiar/scratch_ctx.py`` -> ``agent_clay`` -> ``clay_view``)
    must not come back in :func:`_pure_packages.pure_packages`.

    This is a from-scratch reproduction rather than a read of the real
    ``studio/familiar/`` package on purpose: this fix is what moves
    ``apply.py``/``scratch_ctx.py`` out of ``familiar`` entirely, so a test
    that depended on their being there would stop meaning anything the day
    the move landed. The tmp tree keeps the claim -- "a relative import
    chain into a window is not invisible" -- true independent of that move.
    """
    studio = tmp_path / "warlock" / "studio"
    studio.mkdir(parents=True)
    (tmp_path / "warlock" / "__init__.py").write_text("", encoding="utf-8")
    (studio / "__init__.py").write_text("", encoding="utf-8")

    # A leaf that is a window at module scope -- the same shape as the real
    # ``clay_view.py`` (``import moderngl``).
    (studio / "gl_leaf.py").write_text("import moderngl\n", encoding="utf-8")
    # A studio-level module reaching the window only through a relative
    # import -- the same shape as the real ``agent_clay.py``.
    (studio / "bridge.py").write_text("from .gl_leaf import Thing\n", encoding="utf-8")

    # The package under test: reaches ``bridge`` (and so the window) only
    # through ``from .. import bridge`` -- the same shape as the real
    # ``familiar/scratch_ctx.py``'s ``from .. import agent_clay``.
    pkg = studio / "ghost"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "reaches_out.py").write_text("from .. import bridge\n", encoding="utf-8")

    roots = pp._module_roots(pkg / "reaches_out.py")
    assert "moderngl" in roots, (
        "a relative import chain into a window module must surface in the "
        f"resolved roots; got {roots}"
    )


def test_familiar_is_pure_now_that_the_gl_side_moved_to_studio_level():
    """``studio/familiar/`` itself, post-move: it may still import
    ``clay_mode`` (a plain controller module, no window) but no longer
    ``agent_clay`` -- that reaches the window and now lives in
    ``studio/familiar_preview.py`` instead."""
    assert "familiar" in pp.pure_packages()
