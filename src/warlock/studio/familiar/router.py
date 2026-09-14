"""Familiar's router: one short generation that picks one skill.

Stdlib only (no imgui, moderngl, pygame, ``service``, ``queue`` or httpx) --
this module is pure routing logic, called from the frame thread and from
``warlock-agent-host``-style workers alike, and must stay importable from
either without dragging a GL context or an event loop in.

**Slot plan.** ``pipelines/llama.py`` runs the server with
``PARALLEL_SLOTS = 2`` (llama.py:62) and ``CTX_SIZE = 16384`` (llama.py:63):
slot 0 is reserved for the router's own short, constrained generation
(:data:`ROUTE_SCHEMA` against :data:`SKILLS`) and slot 1 for the routed
skill's generation, so a long-running skill reply never blocks the next
message's routing decision behind it in the same slot's queue.

**The frozen router card** (the system prompt and few-shot content that
actually drives the model to emit one of :data:`SKILLS`) is T6's
``contract.CARDS["router"]`` -- ``cards/router-1.txt`` -- built via
:func:`~.contract.build_router_messages`. This module only defines the
schema and parses the result; it stays free of ``contract``'s own frozen-card
machinery (hashing, sampling, message-building) the same way it stays free
of httpx, so a training script or a future router-only test never has to
pull either in.
"""

from __future__ import annotations

import json

#: Every skill the router can send a message to. "other" is the fallback for
#: a request no skill claims, and also what :func:`parse_route` returns when
#: the model's own output cannot be trusted -- one enum, so the two failure
#: modes (the model chose "other"; the model said something unparseable) are
#: indistinguishable to a caller, which is the point: both mean "don't act,
#: just answer".
SKILLS: tuple[str, ...] = (
    "clay_build",
    "clay_edit",
    "manual",
    "character",
    "create",
    "navigate",
    "other",
)

#: The JSON schema handed to llama-server's ``response_format`` for the
#: router's generation -- constrained decoding, so the model can only ever
#: emit one of :data:`SKILLS` rather than free text this module would then
#: have to coerce.
ROUTE_SCHEMA: dict = {
    "type": "object",
    "properties": {"skill": {"enum": list(SKILLS)}},
    "required": ["skill"],
    "additionalProperties": False,
}

#: llama-server slot for the router's short generation. See the module
#: docstring's slot plan.
ROUTER_SLOT = 0

#: llama-server slot for the routed skill's generation. See the module
#: docstring's slot plan.
SKILL_SLOT = 1


def parse_route(text: str) -> str:
    """The skill named in ``text``, or ``"other"`` if it cannot be trusted.

    Never raises: a router call that comes back malformed (truncated,
    fenced, the wrong shape, an enum value the model invented) is not a bug
    in this function to surface, it is exactly the situation :data:`SKILLS`
    already has an answer for. Tolerates a ```` ```json ... ``` ```` fence
    and surrounding whitespace, because constrained decoding still sometimes
    rides inside a code fence in practice.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (with an optional language tag) and the
        # closing fence, keeping whatever sits between them.
        stripped = stripped.lstrip("`")
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        closing = stripped.rfind("```")
        if closing != -1:
            stripped = stripped[:closing]
        stripped = stripped.strip()

    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return "other"

    if not isinstance(parsed, dict):
        return "other"

    skill = parsed.get("skill")
    if skill in SKILLS:
        return skill
    return "other"
