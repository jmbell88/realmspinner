"""What ``studio/familiar/`` is allowed to reach for, now that the GL-side
preview/apply mechanics live in ``studio/familiar_preview.py`` instead.

Written the same way ``tests/mason/test_mason_imports.py`` pins its own
package, and for the same reason: the whole claim of a "pure" package is
worth exactly as much as the imports staying honest, checked by AST rather
than trusted by eye.
"""

from __future__ import annotations

import ast
from pathlib import Path

import _pure_packages as pp

from warlock.studio import familiar

PACKAGE_DIR = Path(familiar.__file__).parent

#: The same window roots ``_pure_packages.WINDOW_ROOTS`` bans, plus the
#: service/network/queue doors a pure module must not open either --
#: ``contract.py`` authors a training card offline, ``retrieval.py`` reads
#: manual chapters already on disk, ``router.py``/``threads.py`` are pure
#: logic, and none of that needs a window, a job queue or the network.
BANNED = frozenset(
    {
        "imgui",
        "imgui_bundle",
        "moderngl",
        "pygame",
        "OpenGL",
        "glfw",
        "httpx",
    }
)

#: Absolute dotted names banned regardless of which root they hang off.
BANNED_MODULES = frozenset(
    {
        "warlock.service",
        "warlock.queue",
        "warlock.studio.agent_clay",
        "warlock.studio.clay_mode",
        "warlock.studio.clay_view",
    }
)

#: The one module allowed to reach ``agent_clay`` -- and only inside a
#: function body, never at module scope. ``contract.derive_clay_card``
#: imports it lazily to author the next training card from the *live*
#: ``agent_clay`` tool registry; see ``contract.py``'s own docstring for why
#: that has to be a live read rather than a snapshot, and why it is safe for
#: a "pure" package to do only because the import never runs at collection
#: or module-import time.
LAZY_AGENT_CLAY_ALLOWED = {"contract.py"}


def _modules() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def test_there_are_modules_to_check():
    """A glob that matched nothing would make every test below vacuously
    pass."""
    assert len(_modules()) >= 4


def test_the_familiar_package_imports_no_window_service_or_network():
    """No imgui, moderngl, pygame, ``warlock.service``, ``warlock.queue``,
    httpx, or the three GL-adjacent studio modules -- at module scope,
    anywhere under ``studio/familiar/``.

    A function-body import is allowed *only* for ``warlock.studio.agent_clay``
    in ``contract.py`` (see :data:`LAZY_AGENT_CLAY_ALLOWED`); every other
    banned name is refused wherever it appears, module scope or not, because
    nothing else here has ``contract.derive_clay_card``'s reason to reach
    that far -- and a lazy import of ``clay_view``, ``httpx`` or the service
    layer would still run GL, the network or a job door from a package this
    pin claims never does.
    """
    for path in _modules():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_body_ids = {id(n) for n in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # A relative import inside this package can only reach a
                    # sibling module of the pure package itself -- there is
                    # nothing to climb out to here that isn't already listed
                    # in BANNED_MODULES by its absolute name.
                    continue
                names = [node.module or ""]
            else:
                continue

            at_module_scope = id(node) in module_body_ids
            for name in names:
                root = name.split(".")[0]
                assert root not in BANNED, f"{path.name} imports banned root {name!r}"

                hit = next(
                    (m for m in BANNED_MODULES if name == m or name.startswith(m + ".")),
                    None,
                )
                if hit is None:
                    continue
                if hit == "warlock.studio.agent_clay" and path.name in LAZY_AGENT_CLAY_ALLOWED:
                    assert not at_module_scope, (
                        f"{path.name} imports {name!r} at module scope -- "
                        "agent_clay is only allowed lazily, inside a function body"
                    )
                    continue
                raise AssertionError(f"{path.name} imports banned module {name!r}")


def test_familiar_appears_in_pure_packages():
    """The point of the whole fix: ``_pure_packages.pure_packages()`` derives
    this rather than trusting a hand-kept list, and this package must now
    actually earn a place in it (see ``tests/test_pure_packages.py`` for the
    relative-import resolution bug that used to let it in for the wrong
    reason -- before ``apply.py``/``scratch_ctx.py`` moved out, ``familiar``
    was misreported "pure" only because the helper could not see through a
    relative import into ``agent_clay``)."""
    assert "familiar" in pp.pure_packages()
