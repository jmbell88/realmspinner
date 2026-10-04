"""MCP prompts for Clay -- pre-written starting points for a driving model,
served over the same RPC v1 pipe as everything else in this package.

**Pure text, no document, no GL.** Rendering a prompt only ever
interpolates its arguments into a template string -- it never touches
``ClayState``, so :meth:`AgentHost._prompt` answers it on the listener
thread directly, the same way :mod:`agent_resources`'s static resources are
answered, and for the same reason.

**Every tool a prompt's rendered text names is checked against the real
tool list**, not assumed to still exist: ``tests/mcp/test_rpc_studio.py``
scans every prompt's rendered text for ``clay_\\w+``/``realmspinner_\\w+``/
``character_\\w+`` tokens and asserts each one is a real tool name from
``agent_clay.tools()`` plus ``agent_character.tools()`` plus
``agent_host.STATUS_TOOL``. The constants below (``_ADD_PRIMITIVE`` and the
rest, plus the ``_CHARACTER_*`` ones the one character-pipeline prompt
uses) exist so a rename of the underlying tool is a one-line fix here
instead of a search-and-replace across every prompt's prose, but the
regression that actually matters is the scan, not the constants -- a prompt
that hand-typed a stale name would still be caught."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# Tool names this module's prompts mention. Not derived from `agent_clay.
# tools()` at import time (that would need a live registry state this leaf
# does not want to require just to build prose) -- named here once, checked
# against the real catalogue by `tests/mcp/test_rpc_studio.py` instead (the
# 2026-09-26 audit, finding agents-clay-03: this comment cited
# `tests/test_agent_prompts.py`, which has never existed), the same
# "derived-not-duplicated, and a test proves it" shape `agent_clay` itself
# uses for its own generator/op catalogues.
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

# The character pipeline's own tools, named here for the identical reason
# the Clay constants above are -- see the module docstring.
_CHARACTER_OPTIONS = "character_options"
_CHARACTER_CREATE = "character_create"
_CHARACTER_JOB = "character_job"
_CHARACTER_SHEET_PREVIEW = "character_sheet_preview"
_CHARACTER_EXPORT = "character_export"


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
    # The 2026-10-03 audit (agents-13): this took "image_path" ("a Library job id
    # or a path/description") and told the model to add it with clay_reference_add,
    # which takes a Library job id or inline base64 and never a path -- the first
    # reference_add refused the input the argument invited. Only the job id
    # survives as an argument; a picture that is not in the Library goes inline
    # to the tool, which the text says.
    job_id = args["job_id"]
    return (
        f"Match the reference picture from Library job {job_id}. Add it with "
        f"{_REFERENCE_ADD} (job_id={job_id!r}; if it is not in the Library, send the "
        f"picture inline as png_base64 instead -- the tool never takes a file path), then "
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


def _character_sheets_from_description(args: dict[str, Any]) -> str:
    description = args["description"]
    movements = args.get("movements")
    # The 2026-10-03 audit's agents-31: this rendered the comma-separated
    # string as bare names ("movements: [walk, idle]"), and character_create
    # takes an array of {"name": ...} objects and refuses a bare name -- a
    # model following the text literally sent a refused first call.
    names = [n.strip() for n in movements.split(",") if n.strip()] if movements else []
    movements_clause = (
        "movements: [" + ", ".join(json.dumps({"name": n}) for n in names) + "]"
        if names
        else "movements omitted, so the resolved species' own default set is used"
    )
    return (
        f"Build a character sheet matching this description: {description}\n\n"
        f"Call {_CHARACTER_OPTIONS} first to see the live vocabulary this "
        f"build ships, then {_CHARACTER_CREATE} with prompt={description!r}, "
        f"{movements_clause}, and directions: 8. It queues both the mesh "
        f"and its rig in one call and returns mesh_job_id and rig_job_id.\n\n"
        f"Poll {_CHARACTER_JOB} on the returned rig_job_id (the mesh's own "
        f"job id reports the same thing too) until its follow_up_sheet_job "
        f"names a job, then poll {_CHARACTER_JOB} on that job until it is "
        f"done. Any terminal status other than done -- error or cancelled -- "
        f"on either job ends the loop. If the rig itself ends in error or is "
        f"cancelled, follow_up_sheet_job never appears -- read "
        f"follow_up_failure (or the rig job's own error) off that same "
        f"{_CHARACTER_JOB} reply and stop, rather than poll forever for a "
        f"sheet that will not come; if the sheet job ends in error or is "
        f"cancelled, read its own error and stop. Look at the finished "
        f"sheet with {_CHARACTER_SHEET_PREVIEW}, then hand it off with "
        f"{_CHARACTER_EXPORT} in each of animated_glb, godot_scene and "
        f"frame_folders. A movement named as part of a 'set' means nothing "
        f"beyond appearing in that movements list -- there is no other "
        f"bookkeeping to it."
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
                    "name": "job_id",
                    "description": "The Library job id of the reference image to match "
                    "(a picture not in the Library cannot be named here; send it inline "
                    "to clay_reference_add instead).",
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
        _Prompt(
            "character_sheets_from_description",
            "Character sheet from a description",
            "Build a character and export it as a sprite sheet from a plain-text "
            "description.",
            [
                {
                    "name": "description",
                    "description": "What to build, in plain language.",
                    "required": True,
                },
                {
                    "name": "movements",
                    "description": "A comma-separated list of movements to render. "
                    "Omit to use the resolved species' own default set.",
                    "required": False,
                },
            ],
            _character_sheets_from_description,
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
    # The 2026-09-26 audit, finding agents-clay-04: MCP prompt arguments are
    # always strings -- there is no other type in the protocol -- but this
    # only ever checked whether a declared name was *present* in `arguments`,
    # never whether its value actually was one. `{"description": None}` used
    # to count as present and sail straight into a render function's own
    # f-string, landing the literal text "None" (and, for a numeric
    # `movements`, its own stray digits) into the served prompt text. A
    # present value of the wrong type is now dropped -- treated exactly like
    # a caller who never sent it -- so a required one still shows up in
    # `missing` below and an optional one falls back to its own render
    # function's `.get(...)` default instead of being stringified.
    # The 2026-10-03 audit's agents-20: an empty or whitespace-only string is
    # a str, so it counted as present -- a required argument satisfied by
    # nothing rendered "a Clay model matching this description: ". Blank is
    # absent, the same as the wrong-typed case.
    present = {
        a["name"]: arguments[a["name"]]
        for a in prompt.arguments
        if a["name"] in arguments
        and isinstance(arguments[a["name"]], str)
        and arguments[a["name"]].strip()
    }
    missing = [a["name"] for a in prompt.arguments if a["required"] and a["name"] not in present]
    if missing:
        return missing
    text = prompt.render(present)
    return prompt.description, [{"role": "user", "content": {"type": "text", "text": text}}]
