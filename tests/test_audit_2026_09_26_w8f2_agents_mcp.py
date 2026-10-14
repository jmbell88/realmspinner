"""Regression tests for the 2026-09-26 audit's wave-6 MCP findings:
a stale test
citation in agent_prompts.py (agents-clay-03), and agent_prompts.render
stringifying a wrong-typed argument straight into served prompt text
(agents-clay-04). The character tool surface's own findings (agents-character-01,
-02 and -04) went with that surface.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from realmspinner.service import rig as svc_rig
from realmspinner.studio import agent_prompts

# --- agents-character-03: what outlived the character tool surface ---


def _finished_mesh_job(svc: Any) -> str:
    job_id = svc.store.create("image", "a knight", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    return job_id


def test_create_rig_still_replaces_an_existing_rig_for_library_and_create(
    svc: Any,
) -> None:
    """Library's "Rig this mesh again" and Create's Rig stage both call this door
    to replace an existing rig on purpose. The ``refuse_existing`` opt-out that
    sat beside that behaviour left with the character agent surface (the
    2026-10-14 audit, finding docs-E-17); this pins what remains."""
    job_id = _finished_mesh_job(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid"}), encoding="utf-8"
    )
    out = svc_rig.create_rig(svc, job_id, template="quadruped")
    assert out["template"] == "quadruped"


# --- agents-clay-03: agent_prompts.py's stale test citation ----------------


def test_agent_prompts_cites_the_real_scan_test_not_a_nonexistent_one() -> None:
    """A comment in agent_prompts.py cited tests/test_agent_prompts.py, which
    has never existed -- the scan it describes
    (test_every_tool_name_a_prompt_mentions_is_a_real_tool) lives in
    tests/mcp/test_rpc_studio.py, which the module's own docstring already
    names correctly two paragraphs above the stale one."""
    import re

    source = Path(inspect.getfile(agent_prompts)).read_text(encoding="utf-8")
    citations = re.findall(r"against the real catalogue by `([^`]+)`", source)
    assert citations, "expected the 'against the real catalogue by ...' comment"
    assert all(c == "tests/mcp/test_rpc_studio.py" for c in citations), citations

    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "tests" / "mcp" / "test_rpc_studio.py").exists()
    assert not (repo_root / "tests" / "test_agent_prompts.py").exists()


# --- agents-clay-04: agent_prompts.render and wrong-typed arguments --------


def test_render_refuses_a_none_valued_required_argument_as_missing() -> None:
    """render() only checked whether a declared name was *present* in
    arguments, never whether its value actually was the string MCP prompt
    arguments always are. {"description": None} used to count as present
    and reach the render function's own f-string, landing the literal text
    "this description: None" in the served prompt. A wrong-typed required
    value is now refused exactly like a missing one."""
    result = agent_prompts.render("model_from_description", {"description": None})
    assert result == ["description"]


def test_render_drops_a_wrong_typed_optional_argument_instead_of_stringifying_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional argument of the wrong type (a count sent as an int, say)
    must not reach a render function's own f-string either -- it is dropped,
    treated exactly like a caller who never sent it, instead of being
    stringified into the served text. No shipped Clay prompt has an optional
    argument, so the rule is proven on a registered stand-in."""
    prompt = agent_prompts._Prompt(
        "probe",
        "Probe",
        "A prompt with one optional argument.",
        [
            {"name": "description", "description": "d", "required": True},
            {"name": "note", "description": "n", "required": False},
        ],
        lambda args: f"{args['description']} / note: {args.get('note', 'omitted')}",
    )
    monkeypatch.setitem(agent_prompts._PROMPTS, "probe", prompt)

    described = agent_prompts.render("probe", {"description": "a barrel", "note": 12345})
    assert described is not None
    _description, messages = described
    text = messages[0]["content"]["text"]
    assert "12345" not in text
    assert "note: omitted" in text
