"""T7's character skill: propose a plan (species, and only whichever of
theme/movements/directions/size/name the user actually asked for), never a
mint. The same split ``doors.py`` already draws between "how to ask the
model" and "what runs once it answers" -- this module only ever builds
messages, builds a JSON schema, and parses a reply string into a plain
``dict``. It never touches ``ctx``, imgui, ``service`` or a registry:
``options`` (every function's own second argument) is a plain dict the
caller (``service.familiar._ask_character``) builds from
``service.characters.character_options`` -- a *service*-layer read this
package is pinned never to make itself
(``tests/familiar/test_familiar_imports.py`` bans ``warlock.service``
wholesale, not just its studio-facing half).

**No frozen card, for ``doors.py``'s own reason.** The list of species Warlock
can build today, and the movements their skeletons ship, is data computed at
call time (a fresh install with fewer downloaded families offers a shorter
list than one with everything installed) -- a frozen few-shot card can name a
fixed vocabulary; it cannot name a vocabulary that changes under it. Both the
system prompt and the offered-options listing here are prose, unversioned,
the same status ``doors.CREATE_SYSTEM``/``NAV_SYSTEM`` already carry.

**The plan is a dry run's input, never its own mint.** :func:`parse_plan`
only ever narrows a reply down to what ``options`` actually offers -- an
unknown movement, an out-of-range size, a theme the species does not paint
are all dropped rather than failing the whole plan, because a caller
(``service.familiar._ask_character``) still owes the *real* refusal to
:func:`~..service.characters.recipe_from_prompt`, which knows the species'
own theme list and the skeleton's own clip library better than a prompt
listing ever could. Never inventing a value the model did not actually name
is the other half of that contract: a plan that echoes back a theme nobody
asked for is a worse answer than one that leaves it unset.
"""

from __future__ import annotations

import json
from typing import Any

CHARACTER_SYSTEM = (
    "You are Familiar's character-drafting assistant. Read the user's "
    "message and the list of species, movements, direction counts and the "
    "size range Warlock can build, then answer with exactly one JSON object "
    "naming a plan: which species to build (or \"none\" if the message "
    "names none of them), and only the theme, movements, direction count, "
    "size or name the user actually asked for. Never invent a theme, a "
    "movement or a name the message did not say -- leave a field out rather "
    "than guess. Never explain your answer, never add prose, never wrap the "
    "JSON in a code fence."
)


def build_character_messages(prompt: str, options: dict[str, Any]) -> list[dict[str, str]]:
    """One ``[system, user]`` turn for ``character``: :data:`CHARACTER_SYSTEM`
    as the system message, *options* (``{"families": [{"key", "label",
    "themes"}], "movements": [...], "directions": [...], "size_range":
    (lo, hi)}``) rendered into the user turn -- a runtime list, **not** part
    of a frozen card, for the module docstring's reason."""
    families = "\n".join(
        f"{f['key']}: {f['label']} (themes: {', '.join(f['themes']) or 'none'})"
        for f in options["families"]
    )
    movements = ", ".join(options["movements"]) or "none"
    directions = ", ".join(str(d) for d in options["directions"])
    lo, hi = options["size_range"]
    return [
        {"role": "system", "content": CHARACTER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Species:\n{families}\n\n"
                f"Movements: {movements}\n"
                f"Directions: {directions}\n"
                f"Size: {lo}-{hi}\n\n"
                f"Message: {prompt}"
            ),
        },
    ]


def character_schema(options: dict[str, Any]) -> dict[str, Any]:
    """The ``response_format`` JSON schema for ``character``: ``family``
    constrained to exactly *options*' own species plus ``"none"``,
    ``movements``/``directions`` constrained to what *options* actually
    offers, ``size`` bounded to *options*' own range -- the same constrained-
    decoding shape :data:`~.doors.navigate_schema`/:func:`~.doors.
    create_schema` already use for their own runtime lists."""
    family_keys = [f["key"] for f in options["families"]]
    lo, hi = options["size_range"]
    return {
        "type": "object",
        "properties": {
            "family": {"enum": [*family_keys, "none"]},
            "theme": {"type": "string"},
            "movements": {
                "type": "array",
                "items": {"enum": list(options["movements"])},
            },
            "directions": {"enum": list(options["directions"])},
            "size": {"type": "integer", "minimum": lo, "maximum": hi},
            "name": {"type": "string", "maxLength": 64},
        },
        "required": ["family"],
        "additionalProperties": False,
    }


def _strip_fence(text: str) -> str:
    """Drop a ```` ```json ... ``` ```` fence and surrounding whitespace, the
    same tolerance :func:`~.doors.parse_target`/:func:`~.router.parse_route`
    apply -- constrained decoding still sometimes rides inside a code fence
    in practice."""
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


def parse_plan(text: str, options: dict[str, Any]) -> dict[str, Any] | None:
    """*text* narrowed to a plan ``dict``, or ``None`` when it names no
    species *options* actually offers -- an explicit ``"none"``, malformed
    JSON, the wrong shape, or a family outside *options*. Never raises, the
    same contract :func:`~.doors.parse_target`/:func:`~.doors.parse_draft`
    keep: a caller gets a clean "don't act" signal, never an exception to
    catch.

    Every other field is **dropped, not fatal**, when it cannot be trusted:
    an unknown movement, a theme the named species does not offer, a
    direction count or size outside *options*' own ladder/range, a blank or
    over-long name -- each one is silently left out of the returned plan
    rather than failing it whole, because the model naming eleven real
    movements and one it invented should not cost the other ten (see the
    module docstring for why the *real* refusal for a value this cannot
    itself validate -- a theme the species truly does not paint -- still
    belongs to ``recipe_from_prompt``, not here).
    """
    family_keys = {f["key"] for f in options["families"]}
    try:
        parsed = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    family = parsed.get("family")
    if not isinstance(family, str) or family not in family_keys:
        return None
    plan: dict[str, Any] = {"family": family}

    themes_by_family = {f["key"]: set(f["themes"]) for f in options["families"]}
    theme = parsed.get("theme")
    if isinstance(theme, str) and theme in themes_by_family.get(family, ()):
        plan["theme"] = theme

    movements_field = parsed.get("movements")
    if isinstance(movements_field, list):
        known = set(options["movements"])
        kept = [m for m in movements_field if isinstance(m, str) and m in known]
        if kept:
            plan["movements"] = kept

    directions = parsed.get("directions")
    # ``bool`` is an ``int`` subclass in Python -- excluded explicitly so a
    # stray ``true``/``false`` in the reply never reads as ``1``/``0``.
    if (
        isinstance(directions, int)
        and not isinstance(directions, bool)
        and directions in options["directions"]
    ):
        plan["directions"] = directions

    size = parsed.get("size")
    lo, hi = options["size_range"]
    if isinstance(size, int) and not isinstance(size, bool) and lo <= size <= hi:
        plan["size"] = size

    name = parsed.get("name")
    if isinstance(name, str) and name.strip() and len(name.strip()) <= 64:
        plan["name"] = name.strip()

    return plan


def plan_overrides(plan: dict[str, Any]) -> dict[str, Any]:
    """*plan* (:func:`parse_plan`'s own shape) mapped onto
    ``service.characters.recipe_from_prompt``'s allowed override keys
    (``_RECIPE_OVERRIDE_KEYS``) -- imported by the *caller's* tests, not
    here, since this module may not import ``service`` even to read a
    frozenset off it.

    Only ``family``/``theme``/``directions``/``name`` ride straight through
    under their own name; ``movements`` becomes ``animations`` (a mapping of
    movement name to ``None``, ``recipe_from_prompt``'s own "resolve this
    clip's own length" shape) and ``size`` becomes ``logical_size`` --
    ``Recipe``'s own field name for a character sheet's pixel size, distinct
    from ``directions``' unrelated "how many facings" meaning."""
    overrides: dict[str, Any] = {"family": plan["family"]}
    if "theme" in plan:
        overrides["theme"] = plan["theme"]
    if "movements" in plan:
        overrides["animations"] = {name: None for name in plan["movements"]}
    if "directions" in plan:
        overrides["directions"] = plan["directions"]
    if "size" in plan:
        overrides["logical_size"] = plan["size"]
    if "name" in plan:
        overrides["name"] = plan["name"]
    return overrides
