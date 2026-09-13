"""MCP prompts for Clay -- pre-written starting points for a driving model,
served over the same RPC v1 pipe as everything else in this package.

**Pure text, no document, no GL.** Rendering a prompt only ever
interpolates its arguments into a template string -- it never touches
``ClayState``, so :meth:`AgentHost._prompt` answers it on the listener
thread directly, the same way :mod:`agent_resources`'s static resources are
answered, and for the same reason.

**Every tool a prompt's rendered text names is checked against the real
tool list**, not assumed to still exist: ``tests/test_agent_prompts.py``
scans every prompt's rendered text for ``clay_\\w+``/``warlock_\\w+`` tokens
and asserts each one is a real tool name from ``agent_clay.tools()`` plus
``agent_host.STATUS_TOOL``. The constants below (``_ADD_PRIMITIVE`` and the
rest) exist so a rename of the underlying tool is a one-line fix here
instead of a search-and-replace across every prompt's prose, but the
regression that actually matters is the scan, not the constants -- a prompt
that hand-typed a stale name would still be caught."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Tool names this module's prompts mention. Not derived from `agent_clay.
# tools()` at import time (that would need a live registry state this leaf
# does not want to require just to build prose) -- named here once, checked
# against the real catalogue by `tests/test_agent_prompts.py` instead, the
# same "derived-not-duplicated, and a test proves it" shape `agent_clay`
# itself uses for its own generator/op catalogues.
_SCENE = "clay_scene"
_ADD_PRIMITIVE = "clay_add_primitive"
_ADD_FIGURE = "clay_add_figure"
_TRANSFORM = "clay_transform"
_SET_PARAMS = "clay_set_params"
_MATERIAL = "clay_material"
_BOOLEAN = "clay_boolean"
_RENDER = "clay_render"
_DIAGNOSE = "clay_diagnose"
_EXPORT = "clay_export"
_ELEMENTS = "clay_elements"
_OP = "clay_op"
_ELEMENT_MODE = "clay_element_mode"
_SELECT_BY = "clay_select_by"
_REFERENCE_ADD = "clay_reference_add"


def _model_from_description(args: dict[str, Any]) -> str:
    description = args["description"]
    return (
        f"Build a Clay model matching this description: {description}\n\n"
        f"Work in the loop Clay's own conventions recommend: block the shape out with "
        f"{_ADD_PRIMITIVE} (or {_ADD_FIGURE} for a posed figure), {_RENDER} from "
        f"three_quarter and front to see it, adjust with {_TRANSFORM}/{_SET_PARAMS}/"
        f"{_MATERIAL}, combine parts with {_BOOLEAN} where the shape needs one solid, "
        f"run {_DIAGNOSE} before calling it finished, then {_EXPORT}."
    )


def _model_from_reference(args: dict[str, Any]) -> str:
    image_path = args["image_path"]
    return (
        f"Match the reference image at {image_path}. Add it with {_REFERENCE_ADD}, then "
        f"block out the shape with {_ADD_PRIMITIVE} and check your progress with "
        f"{_RENDER}'s 'compare' argument against that reference (try compare_mode "
        f"'beside' first, then 'overlay' once the silhouette is close). Iterate with "
        f"{_TRANSFORM}/{_SET_PARAMS}/{_MATERIAL} and {_BOOLEAN} until the two line up, "
        f"then {_DIAGNOSE} and {_EXPORT}."
    )


def _repair_mesh(args: dict[str, Any]) -> str:
    uid = args.get("uid")
    target = f"the object with uid {uid}" if uid else "every visible object"
    return (
        f"Repair {target}. Call {_DIAGNOSE} first -- pass 'select' on a finding to select "
        f"the elements it names, the same way the properties pane's own click handler "
        f"does. From there, {_ELEMENT_MODE} and {_SELECT_BY} put you in the right mode "
        f"with the right elements selected, and {_OP} (fill-hole, merge-by-distance, "
        f"recalc-normals and the rest of its mesh-repair rows) does the fix. Re-run "
        f"{_DIAGNOSE} afterward to confirm the finding is gone, and {_RENDER} to see the "
        f"result before trusting it."
    )


def _prepare_for_export(args: dict[str, Any]) -> str:
    fmt = args.get("format")
    target = f"the {fmt} format" if fmt else "export"
    return (
        f"Prepare this document for {target}. Run {_DIAGNOSE} across every visible object "
        f"and fix what it finds (see the repair_mesh prompt for the loop), confirm the "
        f"scale and grounding with {_SCENE} and {_RENDER}, then call {_EXPORT}. If "
        f"{_EXPORT} refuses, its message names what to fix -- do not retry blind."
    )


class _Prompt:
    __slots__ = ("name", "title", "description", "arguments", "render")

    def __init__(
        self,
        name: str,
        title: str,
        description: str,
        arguments: list[dict[str, Any]],
        render: Callable[[dict[str, Any]], str],
    ) -> None:
        self.name = name
        self.title = title
        self.description = description
        self.arguments = arguments
        self.render = render


_PROMPTS: dict[str, _Prompt] = {
    p.name: p
    for p in (
        _Prompt(
            "model_from_description",
            "Model from a description",
            "Build a Clay model from a plain-text description.",
            [
                {
                    "name": "description",
                    "description": "What to build, in plain language.",
                    "required": True,
                }
            ],
            _model_from_description,
        ),
        _Prompt(
            "model_from_reference",
            "Model from a reference image",
            "Build a Clay model that matches a reference picture.",
            [
                {
                    "name": "image_path",
                    "description": "A Library job id or a path/description of the reference "
                    "image to match.",
                    "required": True,
                }
            ],
            _model_from_reference,
        ),
        _Prompt(
            "repair_mesh",
            "Repair a mesh",
            "Find and fix what clay_diagnose reports, on one object or the whole scene.",
            [
                {
                    "name": "uid",
                    "description": "The object to repair. Omit to repair every visible "
                    "object.",
                    "required": False,
                }
            ],
            _repair_mesh,
        ),
        _Prompt(
            "prepare_for_export",
            "Prepare for export",
            "Diagnose, fix and confirm a document before exporting it.",
            [
                {
                    "name": "format",
                    "description": "The export format this is headed for, if it matters to "
                    "how the document should be prepared.",
                    "required": False,
                }
            ],
            _prepare_for_export,
        ),
    )
}


def list_prompts() -> list[dict[str, Any]]:
    """Every prompt's listing entry -- for the RPC v1 ``prompts`` op, MCP's
    own ``prompts/list``, and the catalogue snapshot."""
    return [
        {
            "name": p.name,
            "title": p.title,
            "description": p.description,
            "arguments": list(p.arguments),
        }
        for p in _PROMPTS.values()
    ]


def render(
    name: str, arguments: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]] | list[str] | None:
    """One prompt, rendered. Three outcomes:

    * ``None`` -- no prompt named *name*.
    * A ``list[str]`` -- *name* exists but *arguments* is missing one or more
      of its required arguments; the list names which.
    * ``(description, messages)`` -- rendered successfully. ``messages`` is
      already MCP's own shape: one ``{"role": "user", "content": {"type":
      "text", "text": ...}}`` block.
    """
    prompt = _PROMPTS.get(name)
    if prompt is None:
        return None
    missing = [a["name"] for a in prompt.arguments if a["required"] and a["name"] not in arguments]
    if missing:
        return missing
    text = prompt.render(arguments)
    return prompt.description, [{"role": "user", "content": {"type": "text", "text": text}}]
