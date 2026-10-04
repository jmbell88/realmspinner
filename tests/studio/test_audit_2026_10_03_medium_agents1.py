"""Regression tests for the 2026-10-03 audit's agent-surface Medium findings
agents-12 (a recovery rig takes the character's own family template) and
agents-13 (the model_from_reference prompt only offers inputs
clay_reference_add accepts). agents-03..09 landed with the Criticals and
Highs and are pinned in ``test_audit_2026_10_03_agents_character.py`` and
``tests/modes/clay/test_audit_2026_10_03_agents_clay.py``."""

from __future__ import annotations

import re
from typing import Any

import pytest

from realmspinner.studio import agent_character as ac
from realmspinner.studio import agent_prompts

pytestmark = pytest.mark.filterwarnings("ignore")


def _non_humanoid_family() -> Any:
    from realmspinner.characters import family as family_mod

    for fam in family_mod.families().values():
        if fam.template != "humanoid":
            return fam
    pytest.skip("no shipped family rigs on a non-humanoid template")


def _character_mesh(svc: Any, family_key: str) -> str:
    job_id = "abcdef012345"
    svc.store.create(
        "image", "a beast", {"family": family_key}, job_id, stage="model", status="done"
    )
    return job_id


def test_character_rig_defaults_to_the_characters_own_family_template(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from realmspinner.kernels.rig import store
    from realmspinner.service import rig as svc_rig

    fam = _non_humanoid_family()
    job_id = _character_mesh(svc, fam.key)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(store, "read_rig", lambda job_dir: None)
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: None, raising=False)

    def fake_create_rig(svc, jid, *, template=None, refuse_existing=False):
        seen["template"] = template
        return {"id": "abc123abc123", "source_job": jid, "template": template}

    monkeypatch.setattr(svc_rig, "create_rig", fake_create_rig)
    result = ac.call(svc, ac.Session(), "character_rig", {"job_id": job_id})
    assert result["isError"] is False
    assert seen["template"] == fam.template
    # And the reply says which family decided it.
    assert result["structuredContent"]["template_from_family"] == fam.key


def test_character_rig_still_honours_an_explicit_template_over_the_family(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from realmspinner.kernels.rig import store
    from realmspinner.service import rig as svc_rig

    fam = _non_humanoid_family()
    job_id = _character_mesh(svc, fam.key)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(store, "read_rig", lambda job_dir: None)
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: None, raising=False)
    monkeypatch.setattr(
        svc_rig,
        "create_rig",
        lambda svc, jid, *, template=None, refuse_existing=False: seen.update(template=template)
        or {"id": "abc123abc123", "source_job": jid},
    )
    result = ac.call(
        svc, ac.Session(), "character_rig", {"job_id": job_id, "template": "humanoid"}
    )
    assert result["isError"] is False
    assert seen["template"] == "humanoid"
    assert "template_from_family" not in result["structuredContent"]


def test_character_sheet_create_on_an_unrigged_character_rigs_on_its_family_template(
    monkeypatch: pytest.MonkeyPatch, svc: Any
) -> None:
    from realmspinner.service import rig as svc_rig
    from realmspinner.service import troupe as svc_troupe

    fam = _non_humanoid_family()
    job_id = _character_mesh(svc, fam.key)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(svc_rig, "rig_in_flight", lambda svc, jid: None, raising=False)

    def fake_send(svc, jid, **kwargs):
        seen.update(kwargs)
        return {"id": "abc123abc123", "template": kwargs.get("template")}

    monkeypatch.setattr(svc_troupe, "send_to_troupe", fake_send)
    movement = next(iter(ac._enums().movements))
    result = ac.call(
        svc,
        ac.Session(),
        "character_sheet_create",
        {"job_id": job_id, "movements": [{"name": movement}]},
    )
    assert result["isError"] is False
    assert seen["template"] == fam.template
    assert result["structuredContent"]["template_from_family"] == fam.key


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
