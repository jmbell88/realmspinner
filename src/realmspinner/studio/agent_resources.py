"""MCP resources for Clay -- read-only documents an agent can fetch without
spending a tool call, alongside the tool surface :mod:`agent_clay` already
owns.

**Five URIs, two kinds.** ``realmspinner://clay/scene`` and ``realmspinner://clay/
render/last`` are *dynamic*: answering them touches this session's document
(and, for a render, the shared GL viewport), so they must run on the frame
thread through the same job queue every ``clay_*`` tool call already does
(see ``agent_host.AgentHost._read_resource``) -- never on the listener
thread, which is the same rule ``agent_host``'s own module docstring states
for a ``call``. ``realmspinner://clay/conventions``, ``realmspinner://clay/generators``
and ``realmspinner://clay/operations`` are *static*: pure functions of registries
that already exist for a human surface (:mod:`agent_clay`'s own prose,
``clay.primitives.GENERATORS``, ``clay_ops.OPS``), touching no document and
no GL, so they are safe to answer on the listener thread directly and cheap
enough to embed inline in the catalogue snapshot (see
:func:`catalogue_resources`) -- a bridge dialled while Realmspinner itself is not
running can still read them.

**Never hand-listed.** :func:`_generators_json` and :func:`_operations_json`
walk ``primitives.GENERATORS`` and ``clay_ops.OPS`` the same way
``agent_clay._generator_catalog``/``_op_catalog`` already do for their own
prose -- a thirteenth primitive or op needs no edit here either.

**Bound memory, on purpose.** ``Session.last_render_png`` (see ``agent_clay.
Session``) holds at most one PNG -- the most recent ``clay_render`` this
session produced, overwritten by the next one -- never a history. A
connection that never renders costs this nothing; one that renders often
never grows past one image."""

from __future__ import annotations

import json
from typing import Any

SCENE_URI = "realmspinner://clay/scene"
RENDER_LAST_URI = "realmspinner://clay/render/last"
CONVENTIONS_URI = "realmspinner://clay/conventions"
GENERATORS_URI = "realmspinner://clay/generators"
OPERATIONS_URI = "realmspinner://clay/operations"

#: The two URIs :func:`read_dynamic` answers -- everything else static
#: resources cover, and an unknown uri is neither.
DYNAMIC_URIS = frozenset({SCENE_URI, RENDER_LAST_URI})


def _generators_json() -> dict[str, Any]:
    """``primitives.GENERATORS``, as JSON: for each generator, its default
    parameters. Derived, never hand-listed -- see the module docstring."""
    from ..kernels.mesh import primitives as bp

    return {name: {"params": dict(defaults)} for name, (defaults, _fn) in bp.GENERATORS.items()}


def _param_json(param: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"name": param.name, "label": param.label, "default": param.default}
    if param.boolean:
        row["kind"] = "boolean"
    elif param.choices:
        row["kind"] = "choice"
        row["choices"] = list(param.choices)
    else:
        row["kind"] = "number"
        row["low"] = param.low
        row["high"] = param.high
    if param.warn:
        row["warn"] = param.warn
    return row


def _operations_json() -> dict[str, Any]:
    """``clay_ops.OPS``, as JSON: for each op, the modes it applies to and
    its params' metadata. Derived, never hand-listed -- see the module
    docstring."""
    from .modes.clay import ops as clay_ops

    return {
        op.name: {
            "label": op.label,
            "modes": list(op.modes),
            "hint": op.hint,
            "params": [_param_json(p) for p in op.params],
        }
        for op in clay_ops.OPS
    }


def read_static(uri: str) -> tuple[str, bytes] | None:
    """A static resource's ``(mimeType, body)``, or ``None`` if *uri* is not
    one of the three static ones. Safe on the listener thread: touches no
    document, no ``ClayState``, no GL."""
    if uri == CONVENTIONS_URI:
        from .modes.clay.agent import dispatch as agent_clay

        return "text/markdown", agent_clay.instructions().encode("utf-8")
    if uri == GENERATORS_URI:
        return "application/json", json.dumps(_generators_json()).encode("utf-8")
    if uri == OPERATIONS_URI:
        return "application/json", json.dumps(_operations_json()).encode("utf-8")
    return None


def read_dynamic(ctx: Any, session: Any, uri: str) -> tuple[str, bytes] | None:
    """A dynamic resource's ``(mimeType, body)``, or ``None`` for "nothing to
    answer with" (no document yet, or no render taken yet). **Frame thread
    only** -- see the module docstring.

    ``realmspinner://clay/scene`` is answered by running the ``clay_scene`` tool
    itself and lifting its ``structuredContent`` back out, rather than
    duplicating ``_h_scene``'s own body here -- one function that knows how
    to describe the scene, not two that can drift apart. A session with no
    document yet gets ``None`` (``not_found``), the same refusal
    ``clay_scene`` itself gives an agent that asks the tool with nothing to
    describe."""
    if uri == SCENE_URI:
        from .modes.clay.agent import dispatch as agent_clay

        result = agent_clay.call(ctx, session, "clay_scene", {})
        if result.get("isError"):
            return None
        return "application/json", json.dumps(result.get("structuredContent")).encode("utf-8")
    if uri == RENDER_LAST_URI:
        png = getattr(session, "last_render_png", None)
        if png is None:
            return None
        return "image/png", png
    return None


def catalogue_resources() -> list[dict[str, Any]]:
    """Every resource's full listing entry, for the RPC v1 ``catalogue`` op
    and the ``<home>/mcp.catalogue.json`` snapshot. The three static entries
    carry their own content inline (``text``), so a bridge dialled while
    Realmspinner is unreachable can still answer ``resources/read`` for them; the
    two dynamic ones carry only metadata, since their content depends on a
    document this snapshot cannot see. :func:`list_resources` is the same
    list with the inline content stripped, for the RPC v1 ``resources`` op
    and MCP's own ``resources/list`` -- a listing is metadata, not a copy of
    every resource's own bytes."""
    conv_mime, conv_body = read_static(CONVENTIONS_URI)  # type: ignore[misc]
    gen_mime, gen_body = read_static(GENERATORS_URI)  # type: ignore[misc]
    ops_mime, ops_body = read_static(OPERATIONS_URI)  # type: ignore[misc]
    return [
        {
            "uri": SCENE_URI,
            "name": "clay-scene",
            "title": "Clay scene",
            "description": "This session's document: every object, the selection, and the "
            "material palette -- the same JSON clay_scene returns.",
            "mimeType": "application/json",
        },
        {
            "uri": RENDER_LAST_URI,
            "name": "clay-render-last",
            "title": "Last Clay render",
            "description": "The most recent picture this session's clay_render produced. "
            "Not found until the first render.",
            "mimeType": "image/png",
        },
        {
            "uri": CONVENTIONS_URI,
            "name": "clay-conventions",
            "title": "Clay conventions",
            "description": "Units, axes, undo and the working loop -- the same prose "
            "agent_clay.instructions() puts in the catalogue.",
            "mimeType": conv_mime,
            "text": conv_body.decode("utf-8"),
        },
        {
            "uri": GENERATORS_URI,
            "name": "clay-generators",
            "title": "Clay generators",
            "description": "Every primitive clay_add_primitive can build, and its default "
            "parameters.",
            "mimeType": gen_mime,
            "text": gen_body.decode("utf-8"),
        },
        {
            "uri": OPERATIONS_URI,
            "name": "clay-operations",
            "title": "Clay operations",
            "description": "Every op clay_op can run, the modes it applies to, and its "
            "parameters.",
            "mimeType": ops_mime,
            "text": ops_body.decode("utf-8"),
        },
    ]


def list_resources() -> list[dict[str, Any]]:
    """:func:`catalogue_resources`, metadata only -- see that function's own
    docstring for why the inline content is stripped here."""
    return [
        {k: v for k, v in r.items() if k not in ("text", "blob")} for r in catalogue_resources()
    ]
