"""The router's schema and its tolerance for a malformed generation.

A test's name is its claim: every failure shape here (garbage text, a
plausible-looking but wrong enum value, a missing key, a non-object) must
fall back to "other" without raising, because a router call sits on the
critical path of every message and a parse exception there would take the
whole assistant down over one bad token.
"""

from __future__ import annotations

import json

from realmspinner.familiar import router


def test_an_unparseable_route_falls_back_to_other():
    assert router.parse_route("not json at all") == "other"
    assert router.parse_route(json.dumps({"skill": "flying_saucer"})) == "other"
    assert router.parse_route(json.dumps({"not_skill": "clay_build"})) == "other"
    assert router.parse_route(json.dumps(["clay_build"])) == "other"
    assert router.parse_route("") == "other"


def test_a_valid_route_is_returned():
    bare = json.dumps({"skill": "clay_build"})
    assert router.parse_route(bare) == "clay_build"

    fenced = f"```json\n{json.dumps({'skill': 'manual'})}\n```"
    assert router.parse_route(fenced) == "manual"

    fenced_no_lang = f"```\n{json.dumps({'skill': 'navigate'})}\n```"
    assert router.parse_route(fenced_no_lang) == "navigate"

    padded = f"  \n{json.dumps({'skill': 'other'})}\n  "
    assert router.parse_route(padded) == "other"


def test_the_route_schema_enum_is_exactly_the_skills():
    assert router.ROUTE_SCHEMA["properties"]["skill"]["enum"] == list(router.SKILLS)
    assert router.ROUTE_SCHEMA["required"] == ["skill"]
    assert router.ROUTE_SCHEMA["additionalProperties"] is False


def test_the_slot_plan_gives_the_router_and_the_skill_different_slots():
    """The router's short generation and the skill's answer must not queue
    behind each other on the same llama-server slot (PARALLEL_SLOTS = 2,
    pipelines/llama.py:62)."""
    assert router.ROUTER_SLOT == 0
    assert router.SKILL_SLOT == 1
    assert router.ROUTER_SLOT != router.SKILL_SLOT
