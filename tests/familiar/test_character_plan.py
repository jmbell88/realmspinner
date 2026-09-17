"""T7: ``studio/familiar/character_plan.py`` -- the pure half of Familiar's
character skill (build the messages, build the schema, parse a reply into a
plan, map a plan onto ``recipe_from_prompt``'s own override keys).

``warlock.familiar.character_plan`` does not exist on the pre-T7
tree, so every test below fails with an ``ImportError`` before its first
assertion runs against the unmodified code.
"""

from __future__ import annotations

import json

from warlock.familiar import character_plan
from warlock.service.characters import _RECIPE_OVERRIDE_KEYS


def _options() -> dict:
    return {
        "families": [
            {"key": "goblin", "label": "Goblin", "themes": ["swamp", "cave"]},
            {"key": "knight", "label": "Knight", "themes": ["steel", "gilded"]},
        ],
        "movements": ["idle", "walk", "run", "jump"],
        "directions": [1, 4, 8, 16],
        "size_range": (8, 256),
    }


# --- schema ------------------------------------------------------------


def test_the_schema_offers_exactly_the_families_plus_none():
    options = _options()
    schema = character_plan.character_schema(options)
    assert set(schema["properties"]["family"]["enum"]) == {"goblin", "knight", "none"}
    assert schema["required"] == ["family"]
    assert schema["additionalProperties"] is False


def test_build_character_messages_lists_species_movements_directions_and_size():
    options = _options()
    messages = character_plan.build_character_messages("make a goblin", options)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "goblin: Goblin" in user
    assert "idle" in user and "jump" in user
    assert "8-256" in user
    assert "make a goblin" in user


# --- parse_plan ----------------------------------------------------------


def test_a_plan_naming_a_species_not_offered_is_none():
    options = _options()
    reply = json.dumps({"family": "dragon"})
    assert character_plan.parse_plan(reply, options) is None


def test_a_plan_naming_none_is_none():
    options = _options()
    reply = json.dumps({"family": "none"})
    assert character_plan.parse_plan(reply, options) is None


def test_malformed_json_is_none():
    options = _options()
    assert character_plan.parse_plan("not json at all", options) is None


def test_unknown_movements_are_dropped_not_fatal():
    options = _options()
    reply = json.dumps({"family": "goblin", "movements": ["walk", "fly", "idle"]})
    plan = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert plan["movements"] == ["walk", "idle"]


def test_a_theme_the_named_species_does_not_offer_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "theme": "gilded"})
    plan = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "theme" not in plan


def test_an_out_of_range_size_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "size": 4096})
    plan = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "size" not in plan


def test_a_direction_count_off_the_ladder_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "directions": 7})
    plan = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "directions" not in plan


def test_a_plan_never_invents_a_field_the_reply_did_not_carry():
    """Only ``family`` is required -- a bare family-only reply must not grow
    a theme, movements, directions, size or name from nowhere."""
    options = _options()
    reply = json.dumps({"family": "goblin"})
    plan = character_plan.parse_plan(reply, options)
    assert plan == {"family": "goblin"}


def test_a_fenced_reply_is_tolerated():
    options = _options()
    reply = "```json\n" + json.dumps({"family": "knight", "name": "Sir Roland"}) + "\n```"
    plan = character_plan.parse_plan(reply, options)
    assert plan == {"family": "knight", "name": "Sir Roland"}


def test_a_blank_name_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "name": "   "})
    plan = character_plan.parse_plan(reply, options)
    assert "name" not in plan


# --- plan_overrides --------------------------------------------------------


def test_plan_overrides_use_only_keys_the_recipe_accepts():
    plan = {
        "family": "goblin",
        "theme": "swamp",
        "movements": ["walk", "idle"],
        "directions": 8,
        "size": 64,
        "name": "Grubnak",
    }
    overrides = character_plan.plan_overrides(plan)
    assert set(overrides) <= _RECIPE_OVERRIDE_KEYS
    assert overrides["family"] == "goblin"
    assert overrides["theme"] == "swamp"
    assert overrides["animations"] == {"walk": None, "idle": None}
    assert overrides["directions"] == 8
    assert overrides["logical_size"] == 64
    assert overrides["name"] == "Grubnak"


def test_plan_overrides_omits_fields_the_plan_never_named():
    overrides = character_plan.plan_overrides({"family": "goblin"})
    assert overrides == {"family": "goblin"}
