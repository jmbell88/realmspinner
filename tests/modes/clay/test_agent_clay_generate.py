""""Generate into the current tab" is a UI door, never an agent one.

Clay's agent surface (``studio/modes/clay/agent/``) derives its whole tool
list from ``primitives.GENERATORS``, ``presets.ASSEMBLIES``, ``ops.OPS`` and
``select.QUERIES`` -- see ``dispatch.py``'s own module docstring -- and none
of those name generation at all. These three tests are the other side of that
claim: that deriving from those registries, rather than hand-listing a fifth
door, actually keeps the tool an agent can drive innocent of text2image and
trellis, both today and after a future edit to any of the four.
"""

from __future__ import annotations

import ast
from pathlib import Path

from studio.test_agent_character import _base_package_for_level, _dotted_names

from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as clay_dispatch

del _base_package_for_level  # imported for _dotted_names' own default use only


def test_generate_into_tab_is_not_a_clay_op():
    """The registry :func:`~.ops.OPS` every op-shaped agent tool and the
    context menu both derive from names no "generate"."""
    names = {op.name for op in clay_ops.OPS}
    assert not any(name == "generate" or name.startswith("generate") for name in names)


def test_no_agent_tool_or_enum_names_generate():
    """Every tool ``dispatch.tools()`` actually publishes, and the two
    registries its enums are derived from (a hand-listed enum would be the
    other way a new capability could sneak in with no new import to catch).
    """
    tools = clay_dispatch.tools()
    tool_names = {t.name for t in tools}
    assert tool_names, "tools() returned nothing -- this test is checking nothing"
    assert not any(name == "generate" or "generate" in name for name in tool_names)

    op_names = {op.name for op in clay_ops.OPS}
    batch_names = set(clay_dispatch._HANDLERS) - clay_dispatch.BATCH_EXCLUDED
    for name in (*op_names, *batch_names):
        assert "generate" not in name, name


#: What the agent surface may never import, by the dotted name an import
#: statement resolves to. The four the brief names: the door itself, and the
#: three service functions that would let a hand-listed op reach it without
#: naming this module at all.
_FORBIDDEN_SUFFIXES = (
    ".clay.generate",
    ".create_job",
    ".promote_to_model",
    ".create_generation_request",
)


def _walk(path: Path, package: str) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        for dotted in _dotted_names(node, package):
            lowered = dotted.lower()
            assert not any(lowered.endswith(bad) for bad in _FORBIDDEN_SUFFIXES), (
                path.name,
                dotted,
            )


def test_the_clay_agent_surface_cannot_import_the_generate_door():
    import realmspinner.studio.agent_resources as agent_resources_mod
    import realmspinner.studio.assistant as assistant_pkg
    import realmspinner.studio.modes.clay.agent as clay_agent_pkg

    for path in sorted(Path(clay_agent_pkg.__file__).parent.glob("*.py")):
        _walk(path, "realmspinner.studio.modes.clay.agent")
    for path in sorted(Path(assistant_pkg.__file__).parent.glob("*.py")):
        _walk(path, "realmspinner.studio.assistant")
    _walk(Path(agent_resources_mod.__file__), "realmspinner.studio")
