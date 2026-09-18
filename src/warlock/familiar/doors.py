"""Familiar T8's two skills: navigating the app, and drafting a brief in
Create. Pure prompt/schema/parse logic, the same split ``contract.py``/
``router.py`` already draw between "how to ask the model" and "what runs
once it answers" -- this module only ever builds messages, builds a JSON
schema, and parses a reply string. It never touches ``ctx``, imgui, the
palette or ``create_stages`` -- that acting half is ``studio/assistant/doors.py``,
one level up, for the same reason ``assistant/ui.py``'s own docstring gives
for living outside this package: navigating and drafting both reach studio
machinery (the palette, Settings, Create's form) this package is pinned
never to import (``tests/familiar/test_familiar_imports.py``).

**Neither skill gets a frozen card.** ``router.py``'s own module docstring
explains why the router's card is frozen (the few-shot examples are what a
routing decision has to stay stable against) -- that reasoning does not
carry over here: ``navigate``'s whole *list of places* is built fresh every
call from whatever the palette and Settings offer *this session*, on *this
machine*, in *this mode* -- a fresh install with three modes gated has a
shorter list than one with everything downloaded. A frozen card can name a
fixed set of skills; it cannot name a set that is itself data computed at
call time. Both prompts here are prose, unversioned, the same status
``CHAT_SYSTEM`` already has.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Destination:
    """One place :func:`build_navigate_messages` may offer the model, and
    (at studio level) one place ``familiar_doors.navigate`` may actually
    send the user. ``key`` is what the router names and what the acting
    half dispatches on -- a palette command's own ``key`` (``go:clay``,
    ``manual``, ``tour:first-look``, ...) or ``settings:<category>``; ``label``
    is the sentence the model reads to decide, borrowed verbatim from
    whatever surface it came from (the palette command's own ``label``, or
    ``app_settings.CATEGORIES``' own title) rather than a second wording
    invented here.
    """

    key: str
    label: str


NAV_SYSTEM = (
    "You are Familiar's navigator. Read the user's message and the list of "
    'places you can send them, then answer with exactly one JSON object: '
    '{"target": "<key>"}, using the key of the single best match, or '
    '{"target": "none"} if nothing listed answers the request. Never '
    "explain your answer, never add prose, never wrap the JSON in a code "
    "fence."
)


def build_navigate_messages(
    prompt: str, destinations: Sequence[Destination]
) -> list[dict[str, str]]:
    """One ``[system, user]`` turn for ``navigate``: :data:`NAV_SYSTEM` as the
    system message, every one of *destinations* rendered ``"key: label"`` and
    joined, then *prompt*, as the user message.

    *destinations* is computed at runtime (see the module docstring for why
    that rules out a frozen card), so it is folded into the user turn here
    rather than baked into a system prompt the way :data:`~.router.SKILLS`'s
    few-shot card is."""
    listing = "\n".join(f"{d.key}: {d.label}" for d in destinations)
    return [
        {"role": "system", "content": NAV_SYSTEM},
        {"role": "user", "content": f"Places:\n{listing}\n\nMessage: {prompt}"},
    ]


def navigate_schema(keys: Sequence[str]) -> dict:
    """The ``response_format`` JSON schema for ``navigate`` -- constrained to
    exactly *keys* plus ``"none"``, the same constrained-decoding shape
    :data:`~.router.ROUTE_SCHEMA` already uses for :data:`~.router.SKILLS`."""
    return {
        "type": "object",
        "properties": {"target": {"enum": [*keys, "none"]}},
        "required": ["target"],
        "additionalProperties": False,
    }


def _strip_fence(text: str) -> str:
    """Drop a ```` ```json ... ``` ```` fence and surrounding whitespace, the
    same tolerance :func:`~.router.parse_route` applies -- constrained
    decoding still sometimes rides inside a code fence in practice."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.lstrip("`")
    first_newline = stripped.find("\n")
    if first_newline != -1:
        stripped = stripped[first_newline + 1 :]
    closing = stripped.rfind("```")
    if closing != -1:
        stripped = stripped[:closing]
    return stripped.strip()


def parse_target(text: str, keys: Sequence[str]) -> str | None:
    """The destination key named in *text*, or ``None`` when it cannot be
    trusted -- an explicit ``"none"``, malformed JSON, the wrong shape, or a
    key outside *keys* (the model naming a destination it was never offered,
    or one this call's *keys* no longer includes). Never raises, the same
    contract :func:`~.router.parse_route` keeps: a caller gets a clean
    "don't act" signal, never an exception to catch."""
    try:
        parsed = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    target = parsed.get("target")
    if isinstance(target, str) and target in keys:
        return target
    return None


CREATE_SYSTEM = (
    "You are Familiar's drafting assistant for Create. Read the user's "
    "message and the list of asset types Create can make, then answer with "
    'exactly one JSON object: {"asset_type": "<key>", "prompt": "<a short '
    'prompt describing what to generate>"}. Never explain your answer, '
    "never add prose, never wrap the JSON in a code fence."
)


def build_create_messages(
    prompt: str, asset_types: Sequence[tuple[str, str]]
) -> list[dict[str, str]]:
    """One ``[system, user]`` turn for ``create``: :data:`CREATE_SYSTEM` as
    the system message, every one of *asset_types* (``(key, label)``, e.g.
    ``create_assets.ASSET_TYPE_OPTIONS``) rendered ``"key: label"`` and
    joined, then *prompt*, as the user message."""
    listing = "\n".join(f"{key}: {label}" for key, label in asset_types)
    return [
        {"role": "system", "content": CREATE_SYSTEM},
        {"role": "user", "content": f"Asset types:\n{listing}\n\nMessage: {prompt}"},
    ]


def create_schema(asset_types: Sequence[tuple[str, str]]) -> dict:
    """The ``response_format`` JSON schema for ``create``: an asset type
    constrained to *asset_types*' own keys, plus a free-text prompt capped at
    1000 characters -- generous for a one-line brief, small enough that a
    model that runs on cannot fill the whole reply budget with it."""
    keys = [key for key, _label in asset_types]
    return {
        "type": "object",
        "properties": {
            "asset_type": {"enum": keys},
            "prompt": {"type": "string", "maxLength": 1000},
        },
        "required": ["asset_type", "prompt"],
        "additionalProperties": False,
    }


def parse_draft(
    text: str, asset_types: Sequence[tuple[str, str]]
) -> tuple[str, str] | None:
    """``(asset_type, prompt)`` named in *text*, or ``None`` when it cannot
    be trusted -- malformed JSON, the wrong shape, an asset type outside
    *asset_types*, or a blank prompt. Never raises, the same contract
    :func:`parse_target`/:func:`~.router.parse_route` keep."""
    keys = {key for key, _label in asset_types}
    try:
        parsed = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    asset_type = parsed.get("asset_type")
    prompt = parsed.get("prompt")
    if asset_type not in keys or not isinstance(prompt, str) or not prompt.strip():
        return None
    return asset_type, prompt.strip()
