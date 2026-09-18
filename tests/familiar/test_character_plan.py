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
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is None
    assert dropped == []


def test_a_plan_naming_none_is_none():
    options = _options()
    reply = json.dumps({"family": "none"})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is None
    assert dropped == []


def test_malformed_json_is_none():
    options = _options()
    plan, dropped = character_plan.parse_plan("not json at all", options)
    assert plan is None
    assert dropped == []


def test_unknown_movements_are_dropped_not_fatal():
    options = _options()
    reply = json.dumps({"family": "goblin", "movements": ["walk", "fly", "idle"]})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert plan["movements"] == ["walk", "idle"]


def test_a_theme_the_named_species_does_not_offer_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "theme": "gilded"})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "theme" not in plan


def test_an_out_of_range_size_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "size": 4096})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "size" not in plan


def test_a_direction_count_off_the_ladder_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "directions": 7})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert "directions" not in plan


def test_a_plan_never_invents_a_field_the_reply_did_not_carry():
    """Only ``family`` is required -- a bare family-only reply must not grow
    a theme, movements, directions, size or name from nowhere."""
    options = _options()
    reply = json.dumps({"family": "goblin"})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan == {"family": "goblin"}
    assert dropped == []


def test_a_fenced_reply_is_tolerated():
    options = _options()
    reply = "```json\n" + json.dumps({"family": "knight", "name": "Sir Roland"}) + "\n```"
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan == {"family": "knight", "name": "Sir Roland"}
    assert dropped == []


def test_a_blank_name_is_dropped():
    options = _options()
    reply = json.dumps({"family": "goblin", "name": "   "})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert "name" not in plan
    # A blank string is not really a name the model *named* -- unlike a real
    # rejected value, it stays a silent drop rather than growing a spurious
    # "(not used: )" line on the plan card.
    assert dropped == []


# --- familiar-02: dropped fields are named, not silently discarded --------


def test_an_unknown_movement_is_named_in_the_dropped_list():
    options = _options()
    reply = json.dumps({"family": "goblin", "movements": ["walk", "fly"]})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    fly_drop = {"kind": "movement", "text": "fly", "reason": "not a movement this build offers"}
    assert fly_drop in dropped


def test_an_unoffered_theme_is_named_in_the_dropped_list():
    options = _options()
    reply = json.dumps({"family": "goblin", "theme": "gilded"})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert any(d["kind"] == "theme" and d["text"] == "gilded" for d in dropped)


def test_an_out_of_ladder_direction_count_is_named_in_the_dropped_list():
    options = _options()
    reply = json.dumps({"family": "goblin", "directions": 7})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert any(d["kind"] == "directions" and d["text"] == "7" for d in dropped)


def test_an_out_of_range_size_is_named_in_the_dropped_list():
    options = _options()
    reply = json.dumps({"family": "goblin", "size": 4096})
    plan, dropped = character_plan.parse_plan(reply, options)
    assert plan is not None
    assert any(d["kind"] == "size" and d["text"] == "4096" for d in dropped)


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
