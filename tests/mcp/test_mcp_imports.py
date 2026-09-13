"""What ``src/warlock/mcp/`` is allowed to reach for, pinned exactly.

``warlock/mcp/__init__.py``'s own docstring states the constraint this file
enforces for the first time: "Every module under here (`protocol.py`,
`pipe.py`, `bridge.py`) is stdlib-only and imports nothing else from `warlock`
-- `bridge.py`'s one exception, `warlock.config`, is display-free and does not
pull in pygame, moderngl or torch. That is what lets `warlock mcp` run as a
tiny child process on a machine with no GPU and no window, exactly like
`doctor` and `sweep`". Nothing has ever checked that this stays true; this is
what does.

Modelled closely on ``tests/sirens/test_sirens_imports.py`` (itself the
"fifth instance" of this pin) -- same ``ast.walk``-based scan, same shape of
tests. The one thing that matters more here than it did there: the forbidden
direction for this package is the app reaching back down into it becoming the
*opposite*, this leaf reaching back up into the app.

That direction used to be allowed one way: ``studio/agent_host.py`` imported
``protocol`` lazily, inside methods, to answer bare MCP JSON-RPC directly on
Studio's own pipe. It no longer does -- Studio speaks RPC v1 exclusively now
(``docs/INVARIANTS.md``'s agent paragraph), and nothing under
``warlock.studio`` may import ``warlock.mcp.protocol`` at all, lazily or
otherwise (the second half of this file, below the package's own outward-
import pins, checks the studio side of that same line). A lazy import the
wrong way round would be exactly as easy to write and just as real a
violation, so both scans walk every node in the tree with ``ast.walk``
rather than only ``tree.body`` -- catching a `def f(): import ...` the way a
body-only scan never would -- and ``test_the_scan_would_catch_a_lazy_studio_
import``/``test_the_studio_scan_would_catch_a_lazy_protocol_import`` prove
that against planted modules rather than trusting the implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock import mcp

PACKAGE_DIR = Path(mcp.__file__).parent
PACKAGE = "warlock.mcp"

#: The one ``warlock`` import this package may make, per ``warlock/mcp/
#: __init__.py``'s own docstring, quoted above rather than reinvented here.
ALLOWED_WARLOCK_IMPORTS = {
    ("bridge.py", "warlock.config"),
}

#: Everything ``doctor``/``sweep`` also stay clear of, for the same reason
#: this package does: a machine running `warlock mcp` has no GPU and no
#: window, and any of these roots would need one or the other.
THIRD_PARTY_ROOTS = {
    "numpy", "PIL", "moderngl", "imgui", "imgui_bundle", "pygame", "torch", "OpenGL",
}

#: The three layers a leaf must never reach into -- the app's UI, its
#: business logic, and its job scheduler -- named separately from
#: ``ALLOWED_WARLOCK_IMPORTS`` so a violation reads as "imported the studio"
#: rather than "not the one allowed import".
FORBIDDEN_WARLOCK_ROOTS = {"warlock.studio", "warlock.service", "warlock.queue"}


def _outward(path: Path, *, package: str = PACKAGE) -> set[str]:
    """Every module *path* imports, from anywhere in the file -- module scope
    or nested inside a function -- resolved to an absolute dotted name.

    ``ast.walk``, not ``tree.body``: see the module docstring for why a scan
    that only looked at top-level statements would miss exactly the case that
    matters here. Relative imports at level 1 (``from . import sibling``) are
    left out on purpose, the same way ``tests/sirens/test_sirens_imports.py``
    leaves them out -- a sibling within the same package has not gone
    anywhere, and counting it would drown the real question ("did this leave
    the package?") in noise from imports that never do.

    An absolute ``from a.b import c`` resolves to ``a.b.c``, not the bare
    ``a.b`` -- unlike the sirens scan this is modelled on, which never needed
    the distinction because every outward import in that package is written
    relative (``from ..undo import UndoStack``). This package's own violation
    to catch is exactly the opposite shape, ``from warlock.studio import
    agent_clay``, and losing ``agent_clay`` off the end would make
    ``test_the_scan_would_catch_a_lazy_studio_import`` catch nothing.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.update(f"{node.module}.{alias.name}" for alias in node.names)
                else:
                    found.update(alias.name for alias in node.names)
            elif node.level >= 2:
                base = package.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _modules() -> list[Path]:
    return sorted(PACKAGE_DIR.glob("*.py"))


def test_there_are_modules_to_check():
    assert len(_modules()) >= 5  # __init__.py, protocol.py, pipe.py, bridge.py, rpc.py


def test_the_only_warlock_import_in_the_mcp_package_is_bridges_config():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == ALLOWED_WARLOCK_IMPORTS


def test_no_third_party_root_is_imported_anywhere_in_the_package():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & THIRD_PARTY_ROOTS), f"{path.name} imports {roots & THIRD_PARTY_ROOTS}"


def test_the_studio_service_and_queue_layers_are_never_imported():
    for path in _modules():
        for name in _outward(path):
            for forbidden in FORBIDDEN_WARLOCK_ROOTS:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


def test_the_scan_would_catch_a_lazy_studio_import(tmp_path):
    """The case the module docstring names by hand: a lazy import inside a
    function, the exact shape ``agent_host``'s own (allowed-direction) lazy
    imports of ``protocol`` already take. Planted in a throwaway module
    rather than asserted against a real file, so this proves the *scan*
    catches the shape rather than proving today's ``src/warlock/mcp/`` files
    happen to be clean."""
    planted = tmp_path / "not_actually_in_mcp.py"
    planted.write_text(
        "def f():\n"
        "    from warlock.studio import agent_clay\n"
        "    return agent_clay\n",
        encoding="utf-8",
    )
    found = _outward(planted)
    assert "warlock.studio.agent_clay" in found


def test_every_module_imports():
    from warlock import mcp  # noqa: F401
    from warlock.mcp import bridge, pipe, protocol, rpc  # noqa: F401


# =============================================================================
# The other side of the same line: nothing under ``warlock.studio`` may
# import ``warlock.mcp.protocol`` -- Studio's own pipe answers RPC v1 only
# now (``docs/INVARIANTS.md``'s agent paragraph), and the bare-MCP dispatcher
# that module used to expose was deleted along with the last caller of it in
# ``studio/agent_host.py``. A regression here would be a lazy, function-local
# import exactly as easily as a module-level one, so this scan is the same
# ``ast.walk`` shape as ``_outward`` above, not a narrower ``tree.body`` one.
# =============================================================================

import warlock.studio as _studio  # noqa: E402

STUDIO_PACKAGE_DIR = Path(_studio.__file__).parent


def _studio_modules() -> list[Path]:
    return sorted(STUDIO_PACKAGE_DIR.rglob("*.py"))


def _imports_mcp_protocol(path: Path) -> bool:
    """Whether *path* imports ``warlock.mcp.protocol`` (module or attribute
    access via ``from warlock.mcp import protocol`` / ``from ..mcp import
    protocol`` / ``from .. import mcp`` used as ``mcp.protocol`` is not
    tracked here -- every real call site in this codebase uses one of the
    first two forms, and a rename to dodge this pin would be its own,
    separately obvious tell)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "warlock.mcp.protocol" or alias.name.endswith(".mcp.protocol"):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("warlock.mcp", "mcp") and any(
                alias.name == "protocol" for alias in node.names
            ):
                return True
            if node.module in ("warlock.mcp.protocol", "mcp.protocol"):
                return True
    return False


def test_no_studio_module_imports_warlock_mcp_protocol():
    offenders = [str(p) for p in _studio_modules() if _imports_mcp_protocol(p)]
    assert not offenders, (
        "warlock.studio must speak only RPC v1 to warlock.mcp -- "
        f"these modules still import warlock.mcp.protocol: {offenders}"
    )


def test_the_studio_scan_would_catch_a_lazy_protocol_import(tmp_path):
    """Proven against a planted module, the same way
    ``test_the_scan_would_catch_a_lazy_studio_import`` proves the package's
    own outward-import scan above -- a lazy, function-local ``from ..mcp
    import protocol`` is exactly the shape this used to be, in
    ``agent_host.py``, before it was deleted."""
    planted = tmp_path / "not_actually_in_studio.py"
    planted.write_text(
        "def f():\n"
        "    from ..mcp import protocol\n"
        "    return protocol\n",
        encoding="utf-8",
    )
    assert _imports_mcp_protocol(planted)
