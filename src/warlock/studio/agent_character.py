"""Warlock's character pipeline, over MCP -- the agent tool surface beside
Clay's (``studio/modes/clay/agent/dispatch.py``).

**Inbound only.** Every handler here runs a door already used by a human
pane (``service.characters``/``service.troupe``/``service.rig`` and their
neighbours): it builds a mesh procedurally, rigs it through a Blender
subprocess, and renders a sprite sheet through the ordinary queue. Nothing
in this module starts an SDXL/TRELLIS generation, imports a model, opens a
socket of its own, or touches ``HF_HUB_OFFLINE`` -- the same rule
``CLAUDE.md``'s Agents bullet states for Clay's own surface, and it holds
here for the same reason: an agent may only ask Warlock to do the things a
person could already do by pressing a button.

**No GL.** ``agent_clay`` imports ``moderngl`` (through ``ClayView``) to
answer ``clay_render``; this module never imports ``agent_clay`` for
exactly that reason -- a character tool has no render of its own, so
nothing here needs a GL context, and importing the module that owns one
would pull it in for no benefit. ``fail``/``ok``/``text``/``image_png`` are
this module's own thin wrappers over ``mcp.rpc`` rather than borrowed from
``agent_clay``, and the unknown-argument refusal below is this module's own
copy of the same rule, not a shared function -- two closed, five-line
refusals are cheaper to keep in sync than a shared import that would have to
cross the same GL boundary this module exists to avoid.

**Registries, not a hand-kept menu.** Every enum a schema declares --
families, themes, movements, rig templates, sheet templates, cameras,
colours -- is read fresh off the same registries the human panes read
(``characters.family``, ``kernels.rig``, ``clips``, ``kernels.charsheet``,
``pipelines.pixelize``, ``service.troupe``/``export``/``characters``) every
time :func:`tools` or a handler runs, through :func:`_enums`. A species or a
shipped clip added tomorrow needs no edit here. ``size`` is the one
exception to "enum": since master's 8b091e98, ``service.troupe`` accepts any
whole pixel size in ``TROUPE_CUSTOM_SIZE_RANGE``, so the schema declares
that same range (``_Enums.size_range``) instead of a ladder, and
``charsheet.SIZES`` survives only as guidance text and in the vocabulary
resource.

**Movements are the *shipped* vocabulary, never a user's edited one.**
``cliplib.shipped_clip_library``/``shipped_clip_names`` read only the
package's own ``templates/clips`` tree, not ``data_dir/poser/clips`` --
unlike Poser's editor, which prefers a user's file whole (see
``cliplib.clip_library``'s own docstring). A user who renames a clip in
Poser must not move what tools/list reports the very next connection,
because a *different* agent talking to a *different* user's install would
then see a schema that disagrees with this one for no reason either could
name. ``character_clips`` (the per-template tool) is the door that *does*
read the user's own copy, exactly as Poser's editor does, alongside a
``shipped`` flag naming which shipped clips a user's edit has kept.

**Field aliasing.** A schema's own argument name is sometimes not the
field a door underneath it refuses by -- ``logical_size`` is a pixel door's
word for what this surface calls ``size``, and so on for the four others in
:data:`_FIELD_ALIASES`. :func:`call` maps a caught :class:`ServiceError`'s
``field`` through that table before building the refusal, and drops back to
``job_id`` (if the tool declares one) or to no field at all when the mapped
name still is not one of the tool's own arguments -- the same "never point
at a control that is not there" rule :func:`_mapped_field`'s own docstring
states.

**Validated here, not left to the door.** Clay's own tools leave a value's
shape to the handler underneath (see ``agent_clay``'s and
``tests/test_agent_schemas.py``'s shared rationale) because every Clay
refusal already names the right field. Several doors this surface calls do
not -- a bad enum can surface as a generic ``ValueError`` with no field at
all, or on a field this surface renamed. So every enum and numeric range a
schema here declares is checked in the handler itself, before any door
runs, and refused on the tool's own argument name -- the alias table exists
for what a *door* still gets wrong under a valid-looking call, not for what
a handler should have caught first.
"""

from __future__ import annotations

import difflib
import functools
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..mcp import rpc

log = logging.getLogger(__name__)

Args = dict[str, Any]


@dataclass
class Session:
    """One MCP connection's claim on the character surface.

    ``minted`` is the whole of this connection's blast-radius limit:
    ``character_cancel`` refuses a job id this dict does not hold (see that
    handler), so an agent can act on a job it created but never reach into
    somebody else's. Keyed by job id, valued by job *kind*
    (``"model"``/``"rig"``/``"charsheet"``) -- not by the wire-facing
    ``"role"`` (``"mesh"``/``"rig"``/``"sheet"``) a minting tool's own
    ``"minted"`` payload uses, because a kind is what ``cancel_job`` and a
    human reading the library actually mean by this job, while a role is
    this surface's own vocabulary for "what this represents in the
    pipeline" and the two are not always the same word (a rig row waiting
    to become a sheet is kind ``"rig"``, role ``"sheet"``).
    """

    minted: dict[str, str] = field(default_factory=dict)
    toast: Callable[[str], None] | None = None


def ok(*content: dict, structured: dict | None = None) -> dict:
    return rpc.ok(*content, structured=structured)


def fail(message: str, **extra: Any) -> dict:
    """Thin wrapper over ``rpc.fail`` -- see ``agent_clay.fail``'s own
    docstring for the same ``field`` -> ``recovery="fix_arguments"``
    default; not called here in favour of a private copy because that
    module may not be imported (see the module docstring's "No GL"
    paragraph)."""
    if "recovery" not in extra and extra.get("field"):
        extra["recovery"] = "fix_arguments"
    return rpc.fail(message, **extra)


def text(s: str) -> dict:
    return rpc.text(s)


def image_png(data: bytes) -> dict:
    return rpc.image_png(data)


def _json(payload: Any) -> str:
    return json.dumps(payload, default=str)


#: See the module docstring's "Field aliasing" paragraph. A tool schema's own
#: argument name -> the field name a door underneath it actually refuses by,
#: for the five cases where the two disagree.
_FIELD_ALIASES: dict[str, str] = {
    "logical_size": "size",
    "layout": "movements",
    "animations": "movements",
    "rig_template": "template",
    "elevation": "camera",
}


def _mapped_field(raw_field: str | None, declared: frozenset[str], message: str = "") -> str | None:
    """*raw_field* (a :class:`ServiceError`'s own ``field``), translated to
    one of *declared* -- a tool's own schema properties -- or ``None``.

    Tried in order: the field itself, if the tool happens to declare it
    unchanged; its alias from :data:`_FIELD_ALIASES`, if that is declared
    instead; on a tool that declares both ``job_id`` and ``sheet_id``
    (``character_sheet_preview``, ``character_export``), whichever of the
    two *message* itself names; ``"job_id"`` or, failing that, ``"sheet_id"``,
    if the tool declares one -- pointing at an id a caller can actually act
    on rather than naming nothing; otherwise ``None``, the same "an honest
    refusal that highlights no control" ``agent_clay.fail`` falls back to for
    its own blanket ``except``.

    The id fallback used to run only when *raw_field* was itself non-empty,
    because an early ``if not raw_field: return None`` returned before ever
    reaching it -- the 2026-09-14 audit (agents-06) found that every
    ``check_job_id``/``check_sheet_id`` refusal (``NotFound("no such job")``,
    ``NotFound("no such sheet")``, the commonest character-tool mistake)
    raises with no ``field`` at all, so the documented fallback never ran for
    exactly the refusal it exists for. It is tried whenever *raw_field* did
    not resolve to a declared name, empty or not.
    """
    if raw_field:
        if raw_field in declared:
            return raw_field
        mapped = _FIELD_ALIASES.get(raw_field)
        if mapped and mapped in declared:
            return mapped
    if "job_id" in declared and "sheet_id" in declared:
        lowered = message.lower()
        if "sheet" in lowered:
            return "sheet_id"
        if "job" in lowered:
            return "job_id"
    if "job_id" in declared:
        return "job_id"
    if "sheet_id" in declared:
        return "sheet_id"
    return None


@dataclass(frozen=True)
class _Enums:
    """Every enum a schema declares, read fresh off the live registries --
    see the module docstring's "Registries, not a hand-kept menu" and
    "Movements are the shipped vocabulary" paragraphs. Built by
    :func:`_enums`, called once per :func:`tools` build and once per
    handler call that needs to validate a value -- cheap enough (a handful
    of dict/list reads) to not need caching, and *must not* be cached: a
    camera preset or a family added at runtime has to appear in the very
    next catalogue (``test_a_camera_preset_added_at_runtime_appears_in_the_
    next_catalogue``)."""

    families: tuple[str, ...]
    themes: tuple[str, ...]
    movements: tuple[str, ...]
    rig_templates: tuple[str, ...]
    sheet_templates: tuple[str, ...]
    directions: tuple[int, ...]
    facings: tuple[str, ...]
    cameras: tuple[str, ...]
    sizes: tuple[int, ...]
    size_range: tuple[int, int]
    fps: tuple[int, ...]
    colors: tuple[int, ...]
    outlines: tuple[str, ...]
    reduce_modes: tuple[str, ...]
    formats: tuple[str, ...]
    filters: tuple[str, ...]


def _enums() -> _Enums:
    from ..characters import family as family_mod
    from ..kernels import charsheet
    from ..kernels.rig import cliplib, templates
    from ..pipelines import pixelize
    from ..service import characters as svc_characters
    from ..service import export as svc_export
    from ..service import troupe as svc_troupe

    families = family_mod.families()
    sheet_templates = tuple(cliplib.shipped_clip_templates())
    movements = tuple(
        sorted({name for t in sheet_templates for name in cliplib.shipped_clip_names(t)})
    )
    return _Enums(
        families=tuple(sorted(families)),
        themes=tuple(sorted({t.key for fam in families.values() for t in fam.themes})),
        movements=movements,
        rig_templates=tuple(r["key"] for r in templates.catalog()),
        sheet_templates=sheet_templates,
        directions=tuple(sorted(charsheet.DIRECTION_PRESETS)),
        facings=tuple(charsheet.COMPASS_16),
        cameras=tuple(key for key, _label, _elev in charsheet.CAMERA_PRESETS),
        sizes=tuple(charsheet.SIZES),
        size_range=tuple(svc_troupe.TROUPE_CUSTOM_SIZE_RANGE),
        fps=tuple(charsheet.FPS_CHOICES),
        colors=tuple(svc_troupe.TROUPE_COLOR_CHOICES),
        outlines=tuple(pixelize.OUTLINE_MODES),
        reduce_modes=tuple(pixelize.REDUCE_MODES),
        formats=tuple(svc_export.CHARACTER_EXPORTS),
        filters=tuple(svc_characters.ASSET_FILTERS),
    )


def _range_refusal(value: Any, lo: int, hi: int, field: str) -> dict | None:
    """``None`` when *value* is a real, in-range number; otherwise the
    refusal :func:`fail` builds for it. A ``bool`` is rejected outright --
    ``isinstance(True, int)`` is ``True`` in Python, and a caller sending
    ``true`` where a count belongs is a type mistake, not a boundary one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fail(f"{field} must be a number.", field=field)
    if not (lo <= value <= hi):
        return fail(f"{field} must be between {lo} and {hi}.", field=field)
    return None


def _camera_elevation(camera: str) -> float:
    from ..kernels import charsheet

    for key, _label, elevation in charsheet.CAMERA_PRESETS:
        if key == camera:
            return elevation
    raise KeyError(camera)  # pragma: no cover - guarded by an enum check first


def _clip_rows(template: str, *, shipped_only: bool) -> list[dict[str, Any]]:
    """One template's clips, projected to ``{name, frames, loop,
    duration_ms, provisional}`` -- the shipped library when *shipped_only*,
    otherwise whatever ``clips.clip_timing`` reports (a user's own edit,
    when there is one)."""
    from .. import clips as clips_mod
    from ..kernels.rig import cliplib

    if shipped_only:
        library = cliplib.shipped_clip_library(template)
        timing = clips_mod.shipped_clip_timing(template)
    else:
        library = cliplib.clip_library(template)
        timing = clips_mod.clip_timing(template)
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
    return rows


# --- tools --------------------------------------------------------------------


def tools() -> list[rpc.Tool]:
    """Every tool this surface offers, built fresh from :func:`_enums` --
    see the module docstring. Called once per catalogue build, the same
    cost ``agent_clay.tools`` already accepts for the same reason."""
    from ..kernels import charsheet

    e = _enums()
    id_schema = {"type": "string", "pattern": "^[0-9a-f]{12}$"}

    move_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(e.movements)},
            "frames": {"type": "integer", "minimum": 1, "maximum": charsheet.MAX_FRAMES},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    move_with_directions_schema = {
        "type": "object",
        "properties": {
            **move_schema["properties"],
            "directions": {"type": "integer", "enum": list(e.directions)},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    movements_cap = max(len(e.movements), 1)

    def pix_properties() -> dict[str, Any]:
        size_lo, size_hi = e.size_range
        return {
            # An integer range, not an enum -- since master's 8b091e98 Send
            # to Troupe accepts any logical size in
            # ``service.troupe.TROUPE_CUSTOM_SIZE_RANGE``, not only the
            # preset ladder ``e.sizes`` still lists for guidance. The
            # description keeps that ladder visible to an agent (and repeats
            # the panes' own nearest-neighbour note) without narrowing what
            # the schema actually accepts.
            "size": {
                "type": "integer",
                "minimum": size_lo,
                "maximum": size_hi,
                "description": (
                    f"{size_lo}-{size_hi}px, any whole number; off "
                    f"{list(e.sizes)} is resized nearest-neighbour."
                ),
            },
            "fps": {"type": "integer", "enum": list(e.fps)},
            "camera": {"type": "string", "enum": list(e.cameras)},
            "colors": {"type": "integer", "enum": list(e.colors)},
            "outline": {"type": "string", "enum": list(e.outlines)},
            "reduce_mode": {"type": "string", "enum": list(e.reduce_modes)},
            "dither": {"type": "boolean"},
            "pixel_art": {"type": "boolean"},
            "palette": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$",
            },
            "name": {"type": "string", "minLength": 1, "maxLength": 64},
        }

    return [
        rpc.Tool(
            name="character_options",
            title="Character options",
            description=(
                "Every choice the character tools accept: families and their "
                "themes and channels, the live clip vocabulary per skeleton, "
                "whether rigging works at all, camera presets, sizes, frame "
                "rates, colours, outline and reduce modes, direction counts, "
                "palettes on disk, whether an export folder is configured, "
                "and the export formats a card may offer."
            ),
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        rpc.Tool(
            name="character_assets",
            title="List character assets",
            description=(
                "This library's characters, newest first -- every one built "
                "through character_create, filtered by riggable/rigged/"
                "has_sheets and paged by cursor."
            ),
            schema={
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "enum": list(e.filters)},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "cursor": {
                        "type": "string",
                        "maxLength": 64,
                        "pattern": r"^[0-9]+(\.[0-9]+)?:[0-9a-f]{12}$",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_clips",
            title="One skeleton's clip vocabulary",
            description=(
                "One sheet template's whole clip library -- shipped, or a "
                "user's own edited copy when Poser has saved one -- each "
                "clip's frame count, whether it loops, its duration, and "
                "whether it is one of the clips this build ships."
            ),
            schema={
                "type": "object",
                "properties": {"template": {"type": "string", "enum": list(e.sheet_templates)}},
                "required": ["template"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_create",
            title="Create a character",
            description=(
                "Build a character from a prompt and/or an explicit family, "
                "and queue its rig in the same call. 'movements' overrides "
                "which clips the eventual sheet asks for and how many "
                "frames each gets; omitted, the resolved species' own "
                "default set is used. Returns mesh_job_id and rig_job_id -- "
                "poll character_job on either to watch the chain land."
            ),
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "maxLength": 1000},
                    "family": {"type": "string", "enum": list(e.families)},
                    "theme": {"type": "string", "enum": list(e.themes)},
                    "appearance": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "movements": {
                        "type": "array",
                        "items": move_schema,
                        "minItems": 1,
                        "maxItems": movements_cap,
                    },
                    "directions": {"type": "integer", "enum": list(e.directions)},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                    **pix_properties(),
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_rig",
            title="Rig a character",
            description=(
                "Add a rig to a finished mesh that does not have one yet. "
                "Refuses a mesh that is already rigged -- this tool adds "
                "rigs, it never replaces one -- and a mesh whose rig is "
                "already running."
            ),
            schema={
                "type": "object",
                "properties": {
                    "job_id": id_schema,
                    "template": {"type": "string", "enum": list(e.rig_templates)},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_sheet_create",
            title="Create a character sheet",
            description=(
                "Render a sprite sheet for a mesh -- rigged already, or "
                "not yet (the sheet is then queued to follow the rig this "
                "call also starts). 'movements' is the complete layout: "
                "each entry names a clip this skeleton ships, and may "
                "override its frame count and direction count; 'directions' "
                "is the default for an entry that omits its own."
            ),
            schema={
                "type": "object",
                "properties": {
                    "job_id": id_schema,
                    "movements": {
                        "type": "array",
                        "items": move_with_directions_schema,
                        "minItems": 1,
                        "maxItems": movements_cap,
                    },
                    "directions": {"type": "integer", "enum": list(e.directions)},
                    "template": {"type": "string", "enum": list(e.sheet_templates)},
                    **pix_properties(),
                },
                "required": ["job_id", "movements"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_job",
            title="Character job status",
            description=(
                "One character's own status -- stage, progress, files, its "
                "rig and follow-up sheet, and every sheet rendered from it, "
                "plus the resource URIs a sheet's own atlas and sidecar are "
                "published under. A large atlas's own resource read refuses "
                "with a small JSON error rather than the image -- use "
                "character_sheet_preview for those instead."
            ),
            schema={
                "type": "object",
                "properties": {"job_id": id_schema},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_sheet_preview",
            title="Preview a character sheet",
            description=(
                "A PNG crop of one sheet -- the whole atlas, one movement's "
                "rows, or one movement/direction strip -- fitted under one "
                "RPC frame."
            ),
            schema={
                "type": "object",
                "properties": {
                    "job_id": id_schema,
                    "sheet_id": id_schema,
                    "max_side": {"type": "integer", "minimum": 16, "maximum": 2048},
                    "movement": {"type": "string", "enum": list(e.movements)},
                    "direction": {"type": "string", "enum": list(e.facings)},
                },
                "required": ["job_id", "sheet_id"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_export",
            title="Export a character",
            description=(
                "Copy a character (and, for the two sheet-shaped formats, "
                "one of its sheets) into WARLOCK_EXPORT_DIR. Refuses "
                "outright when no export folder is configured. Exports land "
                "in a folder named for the asset and its ids, so an agent "
                "export never overwrites an export it did not make."
            ),
            schema={
                "type": "object",
                "properties": {
                    "job_id": id_schema,
                    "format": {"type": "string", "enum": list(e.formats)},
                    "sheet_id": id_schema,
                },
                "required": ["job_id", "format"],
                "additionalProperties": False,
            },
        ),
        rpc.Tool(
            name="character_cancel",
            title="Cancel a character job",
            description=(
                "Cancel a queued or running job this same connection "
                "started -- character_create's mesh or rig, character_rig's "
                "rig, or character_sheet_create's rig or sheet -- or the "
                "sheet job a rig this connection started has since queued "
                "(a sheet render that only begins once that rig finishes). "
                "Anything else is refused, even a job this same character "
                "the connection did not itself start."
            ),
            schema={
                "type": "object",
                "properties": {"job_id": id_schema},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        ),
    ]


def instructions() -> str:
    """The prose ``agent_host`` appends after Clay's own, in the RPC v1
    ``catalogue`` reply -- see the module docstring for what this surface
    does and does not do."""
    from . import agent_host

    return (
        "Warlock's character pipeline, over MCP, beside Clay's tools. Call "
        "character_options first -- it lists every family, theme, rig "
        "template, sheet template, camera preset, size, palette and export "
        "format this session's tools accept, plus the live clip vocabulary "
        "per skeleton; character_clips gives that same vocabulary for one "
        "template, including a user's own edited library.\n\n"
        "job ids and sheet ids (twelve lowercase hex characters) are the "
        "only addresses. There is no path argument anywhere on this "
        "surface -- character_export writes into WARLOCK_EXPORT_DIR, never "
        "a caller-chosen location, and refuses outright when that is not "
        "configured. Every export lands in its own folder, named for the "
        "asset and the ids involved, so a second export of the same asset "
        "reuses that folder on purpose and an export of a different asset "
        "can never land in (or overwrite) one it did not make.\n\n"
        "The chain is mesh, then rig, then sheet. character_create builds "
        "a mesh and queues its rig in one call and returns mesh_job_id and "
        "rig_job_id; character_sheet_create's own path for an unrigged mesh "
        "queues a rig the same way and returns that same rig_job_id. "
        "character_rig adds a rig to a mesh that does not have one yet and "
        "never replaces an existing one; character_sheet_create asks a "
        "rigged mesh for an animated sheet, and -- the same as "
        "character_rig itself -- refuses a mesh whose rig is still "
        "running (field job_id, recovery wait) rather than queuing behind "
        "it. Nothing here pushes "
        "progress: poll character_job on the rig_job_id (the mesh's own id "
        "reports the same follow_up_sheet_job too) until follow_up_sheet_job "
        "names a job, then poll character_job on that job until it is done. "
        "If the rig itself ends in error, no follow-up sheet is ever "
        "queued -- read follow_up_failure (or the rig job's own error) and "
        "stop, rather than poll forever for a sheet that will not appear. "
        "character_sheet_preview, or a sheet's own resource URI "
        "(character_job lists them), shows one once it is done; a sheet's "
        "atlas resource itself refuses to read for a sheet whose PNG does "
        "not fit one RPC frame -- character_sheet_preview has no such "
        "limit and is the right tool for a large sheet either way.\n\n"
        "A movement is not a fixed five: it is whatever character_options' "
        "clip vocabulary reports for the skeleton in play, and "
        "character_sheet_create's own 'movements' argument is the complete "
        "layout a sheet will contain -- nothing is added behind it.\n\n"
        "This surface starts no inference of its own: a prompt is matched "
        "against species and theme words, never sent to a generator, and "
        "every mesh is built procedurally and rigged by a Blender "
        "subprocess the call blocks on.\n\n"
        "An agent may cancel only a job this same connection minted, or the "
        "sheet job a rig it minted has since queued; character_cancel "
        "refuses by job_id otherwise. A call that outruns this bridge's "
        f"{int(agent_host.CALL_TIMEOUT)}-second timeout is handled exactly "
        f"as it is for Clay's own tools -- see that prose -- and "
        f"{agent_host.STATUS_TOOL} takes the operation id either refusal "
        "names."
    )


@functools.cache
def _schemas() -> dict[str, dict]:
    """Tool name -> its whole schema dict. See ``_allowed_argument_names``'s
    own docstring for why caching a *structural* projection of
    :func:`tools` (property names, ``required`` lists, declared ``type``s --
    never an enum's own *values*) is safe."""
    return {t.name: t.schema for t in tools()}


@functools.cache
def _allowed_argument_names() -> dict[str, frozenset[str]]:
    """Tool name -> its schema's own top-level ``properties`` keys. See
    ``agent_clay._allowed_argument_names``'s own docstring for why caching
    this projection (never the whole catalogue) is safe: a property *name*
    is a literal in :func:`tools`'s own source, never derived from a live
    registry -- only an enum's *values* are."""
    return {name: frozenset(schema.get("properties", {})) for name, schema in _schemas().items()}


def _unknown_argument_refusal(name: str, unknown: list[str], allowed: frozenset[str]) -> dict:
    """Mirrors ``agent_clay._unknown_argument_refusal`` -- see that
    function's own docstring. Kept as this module's own copy rather than a
    shared import for the reason the module docstring's "No GL" paragraph
    gives."""
    parts = []
    for key in sorted(unknown):
        match = difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
        if match:
            parts.append(f"{key!r} (did you mean {match[0]!r}?)")
        else:
            parts.append(f"{key!r}")
    noun = "an argument" if len(parts) == 1 else "arguments"
    legal = ", ".join(sorted(allowed)) if allowed else "none -- it takes no arguments at all"
    message = f"{name} does not take {noun} named {', '.join(parts)}. Legal arguments: {legal}."
    return fail(message, field=sorted(unknown)[0])


def _json_type_ok(value: Any, schema_type: str) -> bool:
    """Whether *value* matches one JSON Schema ``"type"`` word, for the
    handful this surface's own schemas ever declare. A ``bool`` is never an
    ``integer``/``number`` -- see :func:`_range_refusal`'s identical
    exclusion and the reason given there."""
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True  # pragma: no cover - every declared type above is covered


def _missing_required_refusal(name: str, missing: list[str]) -> dict:
    """Mirrors :func:`_unknown_argument_refusal`'s own plain style for the
    opposite shape: a *required* argument the caller never sent at all,
    which would otherwise reach a handler's own ``args["..."]`` and surface
    as a bare ``KeyError`` under :func:`call`'s blanket ``except Exception``
    -- "failed unexpectedly", naming nothing. Names every missing key so a
    caller that omitted two does not need two round trips to learn about
    the second; ``field`` is the first (sorted, so which one is
    deterministic), the same "name exactly one" rule
    ``_unknown_argument_refusal`` already follows."""
    ordered = sorted(missing)
    noun = "an argument" if len(ordered) == 1 else "arguments"
    parts = ", ".join(repr(k) for k in ordered)
    message = f"{name} needs {noun} named {parts}."
    return fail(message, field=ordered[0])


def _structural_refusal(name: str, schema: dict, args: Args) -> dict | None:
    """Refuses a call before any handler runs, for the shapes that would
    otherwise reach a handler's own ``args["job_id"]``/``move["name"]``
    indexing and surface as a raw ``KeyError``/``AttributeError`` under the
    generic "failed unexpectedly" refusal -- a missing *required* top-level
    argument, a top-level argument sent as the wrong JSON type, and an
    array item declared as an object (every ``movements``-shaped argument
    this surface has) that is not one, or is one missing that item schema's
    own ``required`` keys.

    See the module docstring's "Validated here, not left to the door"
    paragraph, run one step earlier than any door or handler: an enum's
    value, a number's range, a movement naming a clip that does not exist
    -- all of that is still the handler's own job, exactly as that
    paragraph states. This only keeps a *structurally* broken call from
    reaching a handler at all.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [key for key in required if key not in args]
    if missing:
        return _missing_required_refusal(name, missing)

    for key, value in args.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, dict):
            continue
        schema_type = prop_schema.get("type")
        if schema_type and not _json_type_ok(value, schema_type):
            return fail(f"{key} must be a {schema_type}.", field=key)
        if schema_type == "array" and isinstance(value, list):
            items_schema = prop_schema.get("items")
            if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                for item in value:
                    if not isinstance(item, dict):
                        return fail(f"each entry of {key} must be an object.", field=key)
                    item_missing = [
                        k for k in items_schema.get("required", []) if k not in item
                    ]
                    if item_missing:
                        parts = ", ".join(repr(k) for k in item_missing)
                        return fail(f"each entry of {key} needs {parts}.", field=key)
    return None


def call(svc: Any, session: Session, name: str, arguments: dict) -> dict:
    """Run one tool. Never raises -- see the module docstring's safety
    claim and ``agent_clay.call``'s identical one."""
    if svc is None:
        return fail("Warlock's service is not available.")
    args = arguments or {}
    handler = HANDLERS.get(name)
    if handler is None:
        return fail(f"no such tool: {name!r}")
    schema = _schemas()[name]
    allowed = _allowed_argument_names()[name]
    unknown = [k for k in args if k not in allowed]
    if unknown:
        return _unknown_argument_refusal(name, unknown, allowed)
    structural = _structural_refusal(name, schema, args)
    if structural is not None:
        return structural
    from ..service.errors import ServiceError

    try:
        return handler(svc, session, args)
    except ServiceError as error:
        mapped = _mapped_field(error.field, allowed, error.message)
        if mapped:
            return fail(error.message, field=mapped)
        return fail(error.message)
    except Exception:
        log.exception("agent character tool %r failed", name)
        return fail(f"{name} failed unexpectedly; see the log.")


# --- handlers -------------------------------------------------------------


def _h_character_options(svc: Any, session: Session, args: Args) -> dict:
    del session, args
    from ..characters import family as family_mod
    from ..kernels import charsheet
    from ..kernels.rig import cliplib
    from ..pipelines import pixelize
    from ..service import export as svc_export
    from ..service import palettes
    from ..service import rig as svc_rig
    from ..service import troupe as svc_troupe

    families = []
    for key, fam in sorted(family_mod.families().items()):
        families.append(
            {
                "key": key,
                "label": fam.label,
                "archetype": fam.archetype,
                "template": fam.template,
                "themes": [t.key for t in fam.themes],
                "channels": [c.key for c in fam.channels],
            }
        )

    clip_vocabulary = {
        template: _clip_rows(template, shipped_only=True)
        for template in cliplib.shipped_clip_templates()
    }

    rig_info = svc_rig.rig_templates(svc)
    payload = {
        "families": families,
        "clip_vocabulary": clip_vocabulary,
        "rig": {
            "available": rig_info["available"],
            "detail": rig_info["detail"],
            "templates": [row["key"] for row in rig_info["templates"]],
        },
        "cameras": [
            {"key": key, "label": label, "elevation": elevation}
            for key, label, elevation in charsheet.CAMERA_PRESETS
        ],
        "sizes": list(charsheet.SIZES),
        "fps": list(charsheet.FPS_CHOICES),
        "colors": list(svc_troupe.TROUPE_COLOR_CHOICES),
        "outlines": list(pixelize.OUTLINE_MODES),
        "reduce_modes": list(pixelize.REDUCE_MODES),
        "directions": sorted(charsheet.DIRECTION_PRESETS),
        "palettes": palettes.available(svc.config),
        "export_dir_configured": svc.config.export_dir is not None,
        "formats": [
            {"key": row.key, "label": row.label, "needs_sheet": row.needs_sheet}
            for row in svc_export.CHARACTER_EXPORTS.values()
        ],
    }
    return ok(text(_json(payload)), structured=payload)


def _h_character_assets(svc: Any, session: Session, args: Args) -> dict:
    del session
    from ..service import characters as svc_characters

    e = _enums()
    kwargs: dict[str, Any] = {}
    if "filter" in args:
        if args["filter"] not in e.filters:
            return fail(f"{args['filter']!r} is not an asset filter.", field="filter")
        kwargs["filter"] = args["filter"]
    if "limit" in args:
        refusal = _range_refusal(args["limit"], 1, 50, "limit")
        if refusal:
            return refusal
        kwargs["limit"] = args["limit"]
    if "cursor" in args:
        kwargs["cursor"] = args["cursor"]
    result = svc_characters.list_character_assets(svc, **kwargs)
    return ok(text(_json(result)), structured=result)


def _h_character_clips(svc: Any, session: Session, args: Args) -> dict:
    del session
    from .. import clips as clips_mod
    from ..kernels.rig import cliplib
    from ..service import clips as svc_clips

    template = args["template"]
    library = svc_clips.library(svc, template)
    shipped_names = set(cliplib.shipped_clip_names(template))
    timing = clips_mod.clip_timing(template)
    rows = []
    for clip in library["clips"]:
        name = str(clip["name"])
        clip_time = timing.get(name)
        rows.append(
            {
                "name": name,
                "frames": clip_time.frames if clip_time else None,
                "loop": clip_time.loop if clip_time else None,
                "duration_ms": clip_time.duration_ms if clip_time else None,
                "provisional": bool(clip.get("provisional", False)),
                "shipped": name in shipped_names,
            }
        )
    payload = {"template": template, "clips": rows, "edited": library["edited"]}
    return ok(text(_json(payload)), structured=payload)


def _h_character_create(svc: Any, session: Session, args: Args) -> dict:
    from ..kernels import charsheet
    from ..service import characters as svc_characters

    e = _enums()
    prompt = str(args.get("prompt") or "")
    overrides: dict[str, Any] = {}

    if "family" in args:
        if args["family"] not in e.families:
            return fail(f"{args['family']!r} is not a family.", field="family")
        overrides["family"] = args["family"]
    if "theme" in args:
        if args["theme"] not in e.themes:
            return fail(f"{args['theme']!r} is not a theme.", field="theme")
        overrides["theme"] = args["theme"]
    if "appearance" in args:
        overrides["appearance"] = args["appearance"]

    movements = args.get("movements")
    if movements is not None:
        if not movements:
            return fail("movements must name at least one clip.", field="movements")
        seen: set[str] = set()
        for move in movements:
            move_name = move.get("name")
            if move_name not in e.movements:
                return fail(
                    f"{move_name!r} is not a movement any shipped skeleton offers.",
                    field="movements",
                )
            if move_name in seen:
                return fail(f"two movements are named {move_name!r}.", field="movements")
            seen.add(move_name)
            if "frames" in move:
                refusal = _range_refusal(move["frames"], 1, charsheet.MAX_FRAMES, "movements")
                if refusal:
                    return refusal
        overrides["animations"] = {m["name"]: m.get("frames") for m in movements}

    if "seed" in args:
        refusal = _range_refusal(args["seed"], 0, 2147483647, "seed")
        if refusal:
            return refusal

    if "directions" in args:
        if args["directions"] not in e.directions:
            return fail(
                f"directions must be one of {list(e.directions)}.", field="directions"
            )
        overrides["directions"] = args["directions"]
    if "seed" in args:
        overrides["seed"] = args["seed"]
    if "camera" in args:
        if args["camera"] not in e.cameras:
            return fail(f"{args['camera']!r} is not a camera preset.", field="camera")
        overrides["camera"] = args["camera"]
    if "size" in args:
        refusal = _range_refusal(args["size"], *e.size_range, "size")
        if refusal:
            return refusal
        overrides["logical_size"] = args["size"]
    if "colors" in args:
        if args["colors"] not in e.colors:
            return fail(f"colors must be one of {list(e.colors)}.", field="colors")
        overrides["colors"] = args["colors"]
    if "outline" in args:
        if args["outline"] not in e.outlines:
            return fail(f"outline must be one of {list(e.outlines)}.", field="outline")
        overrides["outline"] = args["outline"]
    if "reduce_mode" in args:
        if args["reduce_mode"] not in e.reduce_modes:
            return fail(
                f"reduce_mode must be one of {list(e.reduce_modes)}.", field="reduce_mode"
            )
        overrides["reduce_mode"] = args["reduce_mode"]
    if "dither" in args:
        overrides["dither"] = args["dither"]
    if "palette" in args:
        overrides["palette"] = args["palette"]
    if "pixel_art" in args:
        overrides["pixel_art"] = args["pixel_art"]
    if "fps" in args:
        if args["fps"] not in e.fps:
            return fail(f"fps must be one of {list(e.fps)}.", field="fps")
        overrides["fps"] = args["fps"]
    if "name" in args:
        overrides["name"] = args["name"]

    built = svc_characters.recipe_from_prompt(svc, prompt, overrides=overrides)
    result = svc_characters.create_character(
        svc,
        built["recipe"],
        name=args.get("name"),
        prompt=prompt,
        resolution=built["resolution"],
    )
    mesh_id, rig_id, kind = result["id"], result["rig"], result["kind"]
    session.minted[mesh_id] = "model"
    session.minted[rig_id] = "rig"
    if session.toast:
        session.toast(f"Created {kind} {mesh_id}")
    payload = {
        "mesh_job_id": mesh_id,
        "rig_job_id": rig_id,
        "kind": kind,
        "recipe": built["recipe"],
        "resolution": built["resolution"],
        "ignored": built["ignored"],
        "cells": built["cells"],
        "estimate_minutes": built["estimate_minutes"],
        "minted": [
            {"job_id": mesh_id, "kind": "model", "role": "mesh"},
            {"job_id": rig_id, "kind": "rig", "role": "rig"},
        ],
    }
    return ok(text(_json(payload)), structured=payload)


def _h_character_rig(svc: Any, session: Session, args: Args) -> dict:
    from ..kernels.rig import store
    from ..service import rig as svc_rig
    from ..service.validation import check_job_id

    e = _enums()
    job_id = args["job_id"]
    template = args.get("template")
    if template is not None and template not in e.rig_templates:
        return fail(f"{template!r} is not a rig template.", field="template")

    check_job_id(job_id)
    svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if store.read_rig(job_dir) is not None:
        return fail("an agent adds rigs; it never replaces one", field="job_id")

    if svc_rig.rig_in_flight(svc, job_id):
        return fail(
            "a rig is already running for this mesh", field="job_id", recovery="wait"
        )

    result = svc_rig.create_rig(svc, job_id, template=template)
    rig_id = result["id"]
    session.minted[rig_id] = "rig"
    if session.toast:
        session.toast(f"Rigging {job_id}")
    payload = {**result, "minted": [{"job_id": rig_id, "kind": "rig", "role": "rig"}]}
    return ok(text(_json(payload)), structured=payload)


def _h_character_sheet_create(svc: Any, session: Session, args: Args) -> dict:
    from ..kernels import charsheet
    from ..service import palettes
    from ..service import rig as svc_rig
    from ..service import troupe as svc_troupe

    e = _enums()
    job_id = args["job_id"]
    movements = args["movements"]
    if not movements:
        return fail("movements must name at least one clip.", field="movements")

    seen: set[str] = set()
    for move in movements:
        move_name = move.get("name")
        if move_name not in e.movements:
            return fail(
                f"{move_name!r} is not a movement any shipped skeleton offers.",
                field="movements",
            )
        if move_name in seen:
            return fail(f"two movements are named {move_name!r}.", field="movements")
        seen.add(move_name)
        if "frames" in move:
            refusal = _range_refusal(move["frames"], 1, charsheet.MAX_FRAMES, "movements")
            if refusal:
                return refusal

    template = args.get("template")
    if template is not None and template not in e.sheet_templates:
        return fail(f"{template!r} is not a skeleton with clips.", field="template")

    default_directions = args.get("directions")
    if default_directions is not None and default_directions not in e.directions:
        return fail(
            f"directions must be one of {list(e.directions)}.", field="directions"
        )
    for move in movements:
        move_directions = move.get("directions")
        if move_directions is not None and move_directions not in e.directions:
            return fail(
                f"directions must be one of {list(e.directions)}.", field="movements"
            )

    if svc_rig.rig_in_flight(svc, job_id):
        return fail(
            "a rig is already running for this mesh", field="job_id", recovery="wait"
        )

    layout: dict[str, Any] = {"version": 3, "movements": []}
    for move in movements:
        row: dict[str, Any] = {"key": move["name"]}
        if "frames" in move:
            row["frames"] = move["frames"]
        row["directions"] = move.get(
            "directions", default_directions if default_directions is not None else 8
        )
        layout["movements"].append(row)
    if "fps" in args:
        if args["fps"] not in e.fps:
            return fail(f"fps must be one of {list(e.fps)}.", field="fps")
        layout["fps"] = args["fps"]

    elevation = None
    if "camera" in args:
        if args["camera"] not in e.cameras:
            return fail(f"{args['camera']!r} is not a camera preset.", field="camera")
        elevation = _camera_elevation(args["camera"])

    pixel_kwargs: dict[str, Any] = {}
    if "size" in args:
        refusal = _range_refusal(args["size"], *e.size_range, "size")
        if refusal:
            return refusal
        pixel_kwargs["logical_size"] = args["size"]
    if "colors" in args:
        if args["colors"] not in e.colors:
            return fail(f"colors must be one of {list(e.colors)}.", field="colors")
        pixel_kwargs["colors"] = args["colors"]
    if "outline" in args:
        if args["outline"] not in e.outlines:
            return fail(f"outline must be one of {list(e.outlines)}.", field="outline")
        pixel_kwargs["outline"] = args["outline"]
    if "reduce_mode" in args:
        if args["reduce_mode"] not in e.reduce_modes:
            return fail(
                f"reduce_mode must be one of {list(e.reduce_modes)}.", field="reduce_mode"
            )
        pixel_kwargs["reduce_mode"] = args["reduce_mode"]
    if "dither" in args:
        pixel_kwargs["dither"] = args["dither"]
    if "palette" in args:
        if args["palette"] not in palettes.available(svc.config):
            return fail(f"{args['palette']!r} is not a palette on disk.", field="palette")
        pixel_kwargs["palette"] = args["palette"]
    if "pixel_art" in args:
        pixel_kwargs["pixel_art"] = args["pixel_art"]
    if "name" in args:
        pixel_kwargs["name"] = args["name"]

    result = svc_troupe.send_to_troupe(
        svc,
        job_id,
        layout=layout,
        template=template,
        elevation=elevation,
        **pixel_kwargs,
    )
    new_id = result["id"]
    kind = "charsheet" if "sheet_id" in result else "rig"
    session.minted[new_id] = kind
    if session.toast:
        session.toast(f"Queued a character sheet for {job_id}")
    payload = {**result, "minted": [{"job_id": new_id, "kind": kind, "role": "sheet"}]}
    if kind == "rig":
        # The mesh had no rig yet: this call minted one, and the follow-up
        # sheet is queued to render once it lands (see module docstring's
        # "The chain is mesh, then rig, then sheet"). Named explicitly here
        # -- rather than left for a caller to infer from "no sheet_id" --
        # so the polling loop instructions() describes has one field to
        # read, on every call that can queue a rig, not just character_create's.
        payload["rig_job_id"] = new_id
    return ok(text(_json(payload)), structured=payload)


def _h_character_job(svc: Any, session: Session, args: Args) -> dict:
    del session
    from ..service import characters as svc_characters
    from . import agent_character_resources as resources_mod

    job_id = args["job_id"]
    result = svc_characters.character_job(svc, job_id)
    resources: list[str] = []
    for sheet in result.get("sheets") or []:
        sheet_id = sheet.get("sheet_id")
        if sheet_id:
            resources.extend(resources_mod.sheet_uris(job_id, sheet_id).values())
    payload = {**result, "resources": resources}
    return ok(text(_json(payload)), structured=payload)


def _h_character_sheet_preview(svc: Any, session: Session, args: Args) -> dict:
    del session
    from ..service import characters as svc_characters

    if "max_side" in args:
        refusal = _range_refusal(args["max_side"], 16, 2048, "max_side")
        if refusal:
            return refusal
    e = _enums()
    if "movement" in args and args["movement"] not in e.movements:
        return fail(f"{args['movement']!r} is not a movement.", field="movement")
    if "direction" in args and args["direction"] not in e.facings:
        return fail(f"{args['direction']!r} is not a direction.", field="direction")

    max_bytes = (rpc.MAX_FRAME - 64 * 1024) * 3 // 4
    png, meta = svc_characters.sheet_preview_png(
        svc,
        args["job_id"],
        args["sheet_id"],
        max_side=args.get("max_side", 512),
        movement=args.get("movement"),
        direction=args.get("direction"),
        max_bytes=max_bytes,
    )
    return ok(image_png(png), text(_json(meta)))


def _normalise_export(fmt: str, result: Any) -> dict[str, Any]:
    """``export.run_character_export``'s own return shape, normalised to
    ``{"format", "dir", "paths": [str, ...]}`` -- or, for ``animated_glb``
    (whose door is ``export.export_to_folder``, a plain file copy with a
    count rather than a list of names), ``{"format", "dir", "copied"}``.
    See the four doors' own return shapes: ``export_to_folder`` ->
    ``{"copied", "dir", "degraded"}``; ``export_package`` ->
    ``{"png", "json", "dir"}``; ``export_godot``/``export_frames`` -> a
    ``Path`` to the staged directory ``export.staged_tree`` built."""
    if isinstance(result, Mapping) and "copied" in result:
        # The 2026-09-15 audit (agents-02): this branch used to drop
        # ``degraded`` on the floor, so an agent reading ``character_export``
        # for ``animated_glb`` was told a mesh whose normalize step actually
        # failed had exported cleanly. Carry it through whenever the door
        # reported one (even an empty list, so a caller can tell "checked,
        # nothing degraded" from "this door predates the field").
        payload: dict[str, Any] = {
            "format": fmt,
            "dir": result["dir"],
            "copied": result["copied"],
        }
        if "degraded" in result:
            payload["degraded"] = result["degraded"]
        return payload
    if isinstance(result, Mapping):
        paths = [str(v) for k, v in result.items() if k != "dir"]
        return {"format": fmt, "dir": str(result["dir"]), "paths": paths}
    path = Path(result)
    return {"format": fmt, "dir": str(path.parent), "paths": [str(path)]}


def _h_character_export(svc: Any, session: Session, args: Args) -> dict:
    from ..service import characters as svc_characters
    from ..service import export as svc_export

    e = _enums()
    if svc.config.export_dir is None:
        return fail("no export folder configured (set WARLOCK_EXPORT_DIR)")
    fmt = args["format"]
    if fmt not in e.formats:
        return fail(f"{fmt!r} is not a character export format.", field="format")
    job_id = args["job_id"]
    sheet_id = args.get("sheet_id")
    # Named for the asset and the ids actually in play -- never for a job's
    # own (agent- or user-chosen) name -- so a second agent export of the
    # same mesh/sheet pair lands in the same folder on purpose, and an
    # export of a *different* pair can never collide with one it did not
    # make (the scope rule this fix is named for). ``needs_sheet`` decides
    # whether the sheet id is even part of the stem: the two mesh-only
    # formats must not fold a caller-supplied (and unused) sheet_id into it.
    needs_sheet = svc_export.CHARACTER_EXPORTS[fmt].needs_sheet
    stem = svc_characters.agent_export_stem(svc, job_id, sheet_id if needs_sheet else None)
    result = svc_export.run_character_export(svc, fmt, job_id, sheet_id, stem=stem)
    payload = _normalise_export(fmt, result)
    if session.toast:
        session.toast(f"Exported {job_id} as {fmt}")
    return ok(text(_json(payload)), structured=payload)


def _rig_queued_this_sheet(svc: Any, session: Session, job_id: str) -> bool:
    """Whether *job_id* is the follow-up sheet job of a rig this session
    minted -- resolved fresh at call time (never cached on :class:`Session`)
    because the follow-up may not have existed yet when the rig was minted.
    See ``_h_character_cancel``'s own docstring for why this is the one
    exception to "an agent may cancel only the jobs it started"."""
    from ..service import troupe as svc_troupe
    from ..service.errors import ServiceError

    for minted_id, kind in session.minted.items():
        if kind != "rig":
            continue
        try:
            if svc_troupe.follow_up_sheet_job(svc, minted_id) == job_id:
                return True
        except ServiceError:
            continue
    return False


def _h_character_cancel(svc: Any, session: Session, args: Args) -> dict:
    """Cancel a job this connection minted, or -- the one thing it did not
    itself mint that it may still reach -- the follow-up sheet job a rig it
    minted has since queued. That sheet is the direct continuation of a
    press this connection made (see ``troupe.send_to_troupe``'s own
    "one press mints one row, the second is minted by the same mechanism"
    paragraph); refusing to let an agent cancel it would leave a
    caller-visible job in flight that this same connection has no way to
    stop, for no safety this surface's blast-radius rule actually needs."""
    from ..service import _jobs_lifecycle

    job_id = args["job_id"]
    if job_id not in session.minted and not _rig_queued_this_sheet(svc, session, job_id):
        return fail(
            "an agent may cancel only a job it started on this connection, or "
            "the sheet job a rig it started has since queued",
            field="job_id",
        )
    result = _jobs_lifecycle.cancel_job(svc, job_id)
    return ok(text(_json(result)), structured=result)


HANDLERS: dict[str, Callable[[Any, Session, Args], dict]] = {
    "character_options": _h_character_options,
    "character_assets": _h_character_assets,
    "character_clips": _h_character_clips,
    "character_create": _h_character_create,
    "character_rig": _h_character_rig,
    "character_sheet_create": _h_character_sheet_create,
    "character_job": _h_character_job,
    "character_sheet_preview": _h_character_sheet_preview,
    "character_export": _h_character_export,
    "character_cancel": _h_character_cancel,
}
