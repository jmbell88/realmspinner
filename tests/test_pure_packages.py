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
    (mirroring the old ``familiar/scratch_ctx.py`` -> ``agent_clay`` ->
    ``clay_view`` chain) must not come back in
    :func:`_pure_packages.pure_packages`.

    This is a from-scratch reproduction rather than a read of the real
    ``studio/familiar/`` package on purpose: this fix is what moves
    ``apply.py``/``scratch_ctx.py`` out of ``familiar`` entirely -- folded
    into ``studio/assistant/preview.py`` -- so a test that depended on their
    being there would stop meaning anything the day the move landed. The
    tmp tree keeps the claim -- "a relative import chain into a window is
    not invisible" -- true independent of that move.
    """
    studio = tmp_path / "warlock" / "studio"
    studio.mkdir(parents=True)
    (tmp_path / "warlock" / "__init__.py").write_text("", encoding="utf-8")
    (studio / "__init__.py").write_text("", encoding="utf-8")

    # A leaf that is a window at module scope -- the same shape as the real
    # ``studio/modes/clay/ui/view.py`` (``import moderngl``).
    (studio / "gl_leaf.py").write_text("import moderngl\n", encoding="utf-8")
    # A studio-level module reaching the window only through a relative
    # import -- the same shape as the real ``studio/modes/clay/agent/dispatch.py``.
    (studio / "bridge.py").write_text("from .gl_leaf import Thing\n", encoding="utf-8")

    # The package under test: reaches ``bridge`` (and so the window) only
    # through ``from .. import bridge`` -- the same shape as the old
    # ``familiar/scratch_ctx.py``'s ``from .. import agent_clay``, before it
    # was folded into ``studio/assistant/preview.py``.
    pkg = studio / "ghost"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "reaches_out.py").write_text("from .. import bridge\n", encoding="utf-8")

    roots = pp._module_roots(pkg / "reaches_out.py")
    assert "moderngl" in roots, (
        "a relative import chain into a window module must surface in the "
        f"resolved roots; got {roots}"
    )


def test_familiar_left_pure_packages_the_same_day_it_left_studio():
    """``familiar`` no longer answers to :func:`pp.pure_packages` at all --
    not because it stopped being pure, but because P3 of the core-vs-subsystems
    restructure (``dev/RESTRUCTURE.md``) moved it straight out of ``studio/``
    to ``warlock/familiar/``, one layer down (L3, beside ``service`` and
    ``characters``) from the L1 kernels and mode-owned ``studio/`` packages
    this function's docstring says it is for.

    This used to assert the opposite -- that ``"familiar" in
    pp.pure_packages()`` -- back when the fix worth recording here was that a
    *relative* import chain into a window was invisible to the derivation
    (see :func:`test_a_relative_import_of_a_gl_module_is_not_counted_pure`
    above, which still holds and is unaffected by the move). That claim is
    obsolete now for a different reason than the bug it fixed: the function
    this file tests answers "which packages are headless engines a sibling
    mode must not import", and ``familiar`` was never a sibling of ``clay`` or
    ``inker`` in that sense -- it earns its own AST pin instead, in
    ``tests/familiar/test_familiar_imports.py``, which proves the same
    "no window, no service, no network" claim directly rather than through
    membership in a set this function no longer has any reason to include it
    in.
    """
    assert "familiar" not in pp.pure_packages()
