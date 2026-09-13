"""MCP resources for the character pipeline -- read-only documents an agent
can fetch without spending a tool call, alongside ``agent_character``'s tool
surface.

**One static resource, one dynamic family.** ``warlock://character/
vocabulary`` is a pure function of the shipped registries -- the same
numbers ``agent_character.tools()`` builds its enums from -- so it costs
nothing to answer and is safe on the listener thread, exactly like
``agent_resources``'s three static Clay resources. A sheet's own sidecar
and atlas (``warlock://character/sheet/<job>/<sheet>/sidecar.json`` and
``.../atlas.png``) are *dynamic*: answering them reads a file a job has (or
has not yet) written, so they run on the service lane through
``agent_host._run_on_service_job``, the character surface's equivalent of
the frame-thread job queue Clay's own dynamic resources use -- there is no
GL and no document behind them, only a finished job's own files, which is
why they go through the *service* lane rather than the frame one.

**Never hand-listed either.** :func:`_vocabulary_json` walks the same
registries :func:`agent_character._enums` reads, off the *shipped* clip
libraries only -- see that module's own paragraph on why a user's edited
clip library must never move this resource.

**Inbound only, same as ``agent_character`` itself:** nothing here mutates
a job, starts a process, or reaches a model."""

from __future__ import annotations

import json
import re
from typing import Any

from ..mcp import rpc

VOCABULARY_URI = "warlock://character/vocabulary"

SHEET_URI_RE = re.compile(
    r"^warlock://character/sheet/(?P<job>[0-9a-f]{12})/(?P<sheet>[0-9a-f]{12})/"
    r"(?P<part>sidecar\.json|atlas\.png)$"
)

#: The same one-RPC-frame budget ``agent_character``'s own
#: ``character_sheet_preview`` handler already computes as ``max_bytes`` --
#: see that handler's own reasoning -- applied here to a *whole* atlas PNG
#: instead of a fitted crop. A resource read has no wait/task fallback the
#: way a tool call does (``agent_host._read_resource``'s own docstring: a
#: read that cannot be serviced is just reported ``not_found``, never
#: replayed), so an HD atlas over this bound is refused outright by
#: :func:`read_dynamic` rather than handed to the pipe to fail on.
ATLAS_RESOURCE_MAX_BYTES = (rpc.MAX_FRAME - 64 * 1024) * 3 // 4


def sheet_uris(job_id: str, sheet_id: str) -> dict[str, str]:
    """The two resource URIs one sheet publishes -- what
    ``agent_character``'s ``character_job`` handler lists under
    ``"resources"`` for every sheet a mesh has."""
    return {
        "sidecar": f"warlock://character/sheet/{job_id}/{sheet_id}/sidecar.json",
        "atlas": f"warlock://character/sheet/{job_id}/{sheet_id}/atlas.png",
    }


def owns_uri(uri: str) -> bool:
    """Whether this module answers *uri* at all -- the vocabulary resource,
    or a URI matching :data:`SHEET_URI_RE`. ``agent_host``'s own dynamic
    dispatch (see the module docstring) calls this before routing a
    ``read`` to :func:`read_dynamic`, the same way it checks Clay's
    ``agent_resources.DYNAMIC_URIS`` first."""
    return uri == VOCABULARY_URI or bool(SHEET_URI_RE.match(uri))


def _vocabulary_json() -> dict[str, Any]:
    from .. import clips as clips_mod
    from .. import rigging
    from ..characters import family as family_mod
    from ..pipelines import charsheet, pixelize
    from ..service import export as svc_export
    from ..service import troupe as svc_troupe

    movements: dict[str, list[dict[str, Any]]] = {}
    for template in rigging.shipped_clip_templates():
        library = rigging.shipped_clip_library(template)
        timing = clips_mod.shipped_clip_timing(template)
        rows = []
        for clip in library.get("clips", ()):
            name = str(clip["name"])
            clip_time = timing.get(name)
            rows.append(
                {
                    "name": name,
                    "frames": clip_time.frames if clip_time else None,
                    "loop": clip_time.loop if clip_time else None,
                    "duration_ms": clip_time.duration_ms if clip_time else None,
                    "provisional": bool(clip.get("provisional", False)),
                }
            )
        movements[template] = rows

    directions = {
        str(n): [
            {"key": key, "yaw": yaw, "compass": charsheet.compass_name(yaw)}
            for key, yaw in preset
        ]
        for n, preset in charsheet.DIRECTION_PRESETS.items()
    }

    families = family_mod.families()
    return {
        "movements": movements,
        "directions": directions,
        "cameras": [
            {"key": key, "label": label, "elevation": elevation}
            for key, label, elevation in charsheet.CAMERA_PRESETS
        ],
        "sizes": list(charsheet.SIZES),
        "fps": list(charsheet.FPS_CHOICES),
        "colors": list(svc_troupe.TROUPE_COLOR_CHOICES),
        "outlines": list(pixelize.OUTLINE_MODES),
        "reduce_modes": list(pixelize.REDUCE_MODES),
        "formats": list(svc_export.CHARACTER_EXPORTS),
        "families": [
            {
                "key": key,
                "label": fam.label,
                "archetype": fam.archetype,
                "template": fam.template,
                "themes": [t.key for t in fam.themes],
            }
            for key, fam in sorted(families.items())
        ],
        "sheet_uri_pattern": SHEET_URI_RE.pattern,
    }


def read_static(uri: str) -> tuple[str, bytes] | None:
    """The vocabulary resource's ``(mimeType, body)``, or ``None`` for
    anything else -- safe on the listener thread, see the module
    docstring."""
    if uri == VOCABULARY_URI:
        return "application/json", json.dumps(_vocabulary_json()).encode("utf-8")
    return None


def read_dynamic(svc: Any, uri: str) -> tuple[str, bytes] | None:
    """One sheet's sidecar or atlas, or ``None`` for "nothing to answer
    with" -- an unowned uri, or a job/sheet id this service does not (or no
    longer) hold. **Service lane only**, per the module docstring: reads a
    file through the ordinary ``service.sheets`` door, the ``svc`` this
    call needs rather than a frame-thread ``ctx``.

    An atlas over :data:`ATLAS_RESOURCE_MAX_BYTES` is refused rather than
    read: ``agent_host._read_resource`` treats ``None`` as `not_found`,
    which would be dishonestly silent for "this exists but is too big" --
    indistinguishable, on the wire, from a sheet that was never published at
    all. So this returns a small JSON body instead, still a normal
    ``(mimeType, body)`` answer (the host's own read contract has no
    "refused" shape beyond ``not_found``/a real answer, so a body an agent
    can actually read is the only honest option left) -- pointing at
    ``character_sheet_preview``, the tool built for exactly this case and
    without this resource's frame-sized ceiling."""
    match = SHEET_URI_RE.match(uri)
    if not match:
        return None
    job_id, sheet_id, part = match.group("job"), match.group("sheet"), match.group("part")
    from ..service import sheets as svc_sheets
    from ..service.errors import ServiceError

    try:
        if part == "sidecar.json":
            record = svc_sheets.get_sheet(svc, job_id, sheet_id)
            return "application/json", json.dumps(record, default=str).encode("utf-8")
        path = svc_sheets.sheet_png(svc, job_id, sheet_id)
        size = path.stat().st_size
        if size > ATLAS_RESOURCE_MAX_BYTES:
            error = {
                "error": "atlas too large to read as a resource",
                "size": size,
                "max_bytes": ATLAS_RESOURCE_MAX_BYTES,
                "use": "character_sheet_preview",
            }
            return "application/json", json.dumps(error).encode("utf-8")
        return "image/png", path.read_bytes()
    except ServiceError:
        return None


def catalogue_resources() -> list[dict[str, Any]]:
    """The vocabulary resource's full listing entry, inline content and
    all -- see ``agent_resources.catalogue_resources``'s own docstring for
    why a static resource carries its bytes inline in the catalogue
    snapshot. Sheet resources carry no catalogue entry of their own: unlike
    the three static Clay documents, there is no fixed set of them to list
    ahead of time -- a sheet's own URIs are only ever handed out by
    ``character_job``, the same way Clay's own dynamic resources
    (``warlock://clay/scene``, ``.../render/last``) still get a fixed
    listing row despite depending on a document that may not exist yet.
    """
    mime, body = read_static(VOCABULARY_URI)  # type: ignore[misc]
    return [
        {
            "uri": VOCABULARY_URI,
            "name": "character-vocabulary",
            "title": "Character vocabulary",
            "description": (
                "Every family, movement, direction, camera, size, colour "
                "and export format the character tools accept, off the "
                "shipped registries only -- the same numbers "
                "character_options reports. A sheet's own atlas resource "
                "(published under warlock://character/sheet/<job>/<sheet>/"
                "atlas.png) reads as a small JSON error, not the image, "
                "once the PNG is too large for one RPC frame -- use "
                "character_sheet_preview for a large sheet instead."
            ),
            "mimeType": mime,
            "text": body.decode("utf-8"),
        }
    ]


def list_resources() -> list[dict[str, Any]]:
    """:func:`catalogue_resources`, metadata only -- see
    ``agent_resources.list_resources``'s identical reason."""
    return [
        {k: v for k, v in r.items() if k not in ("text", "blob")}
        for r in catalogue_resources()
    ]
