"""Regression test for the 2026-10-03 audit's agent-surface Medium finding
agents-13 (the model_from_reference prompt only offers inputs
clay_reference_add accepts). agents-12, a recovery rig taking the character's
own family template, went with the character tool surface; agents-03..09 landed
with the Criticals and Highs and are pinned in
``tests/modes/clay/test_audit_2026_10_03_agents_clay.py``."""

from __future__ import annotations

import re

import pytest

from realmspinner.studio import agent_prompts

pytestmark = pytest.mark.filterwarnings("ignore")


def test_model_from_reference_prompt_only_offers_inputs_clay_reference_add_accepts() -> None:
    """agents-13: the argument invited "a path/description of the reference
    image", but clay_reference_add takes a Library job id or inline base64,
    never a path."""
    listed = {p["name"]: p for p in agent_prompts.list_prompts()}["model_from_reference"]
    (argument,) = listed["arguments"]
    assert "path" not in argument["name"]
    assert "path" not in argument["description"].lower()
    assert "job id" in argument["description"].lower()
    rendered = agent_prompts.render("model_from_reference", {argument["name"]: "abcdef012345"})
    assert isinstance(rendered, tuple)
    text = rendered[1][0]["content"]["text"]
    assert "reference image at" not in text
    assert re.search(r"clay_reference_add\W+job_id", text)
    assert "abcdef012345" in text
